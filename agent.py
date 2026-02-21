import os
import pandas as pd

def discover_models(root_path="."):
    """
    Scans a directory to discover various AI model files and pipelines.
    Returns a pandas DataFrame with the results.
    """
    discovered_artifacts = []
    model_types = {
        '.onnx': 'ONNX',
        '.engine': 'TensorRT',
        '.plan': 'TensorRT',
        '.safetensors': 'SafeTensors',
        '.ckpt': 'Checkpoint',
        '.gguf': 'GGUF',
        '.ggml': 'GGML',
        'model_index.json': 'Diffusers Pipeline'
    }

    for root, dirs, files in os.walk(root_path):
        # Rule for Diffusers pipelines
        if 'model_index.json' in files:
            if all(d in dirs for d in ['unet', 'scheduler']): # text_encoder is not always present
                full_path = os.path.join(root, 'model_index.json')
                rel_path = os.path.relpath(full_path, root_path)
                discovered_artifacts.append({
                    "Type": "Diffusers Pipeline",
                    "Path": rel_path,
                    "Details": f"Found pipeline directory at {os.path.dirname(rel_path)}"
                })
                dirs[:] = []  # Don't scan subdirectories of a detected pipeline
                continue

        for file in files:
            file_ext = os.path.splitext(file)[1]
            if file_ext in model_types:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, root_path)
                model_type = model_types[file_ext]
                details = f"File size: {os.path.getsize(full_path) / (1024*1024):.2f} MB"

                if file_ext == '.safetensors':
                    if 'lora' in rel_path.lower():
                        model_type = 'LoRA Adapter'
                    elif 'vae' in rel_path.lower():
                        model_type = 'VAE'
                    else:
                        # A common default assumption, can be refined
                        model_type = 'Stable Diffusion Model'

                discovered_artifacts.append({
                    "Type": model_type,
                    "Path": rel_path,
                    "Details": details
                })

    if not discovered_artifacts:
        return pd.DataFrame(columns=["Type", "Path", "Details"])

    return pd.DataFrame(discovered_artifacts)

if __name__ == '__main__':
    # For testing the discovery function
    # When run from 'app', scans the parent project directory.
    project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    df_models = discover_models(project_root)
    print(df_models.to_string())


def analyze_artifacts(df):
    """
    Analyzes the discovered artifacts and suggests actions.
    Returns the DataFrame with a new 'Suggested Action' column.
    """
    if df.empty:
        df['Suggested Action'] = []
        return df

    actions = []
    for index, row in df.iterrows():
        action = "None"
        model_type = row['Type']
        model_path = row['Path']

        # Rule for suggesting quantization or format conversion
        is_quantized = any(q_word in model_path.lower() for q_word in ['gguf', 'ggml', 'int8', 'int4', 'quantized'])
        is_optimized = any(o_word in model_path.lower() for o_word in ['.engine', '.plan', '.onnx'])
        
        can_be_quantized = ('Checkpoint' in model_type or 'Stable Diffusion Model' in model_type) and not is_quantized
        can_be_converted = 'Diffusers Pipeline' in model_type and not is_optimized

        if can_be_quantized:
            action = "Quantize to GGUF"
        elif can_be_converted:
            action = "Convert to ONNX"

        actions.append(action)

    df['Suggested Action'] = actions
    return df


def execute_agent_action(model_path, action, log_fn, user_prompt=None):
    """
    Executes a suggested action from the agent, or a custom action from a user prompt.
    """
    from gguf_quantizer import quantize_to_gguf
    from convert_onnx import convert_to_onnx_from_diffusers
    from core.local_llm import get_local_llm_client
    
    # Load agent's LLM configuration
    try:
        from gradio_ui import load_config
        config = load_config().get("agent_llm", {})
        llm_client = get_local_llm_client(
            provider=config.get("provider"),
            api_key=config.get("api_key"),
            # Assuming OpenAI compatible structure for local LLMs for now
            base_url=config.get("endpoint_url"),
            model_name=config.get("model_name") 
        )
    except (ImportError, FileNotFoundError):
        llm_client = None
        log_fn("Warning: Could not load LLM client configuration.")

    # If there's a user prompt, the agent's goal is to respond to it.
    if user_prompt:
        log_fn(f"INFO: Agent received instructions: '{user_prompt}' for model: '{model_path}'")
        if not llm_client:
            log_fn("ERROR: Cannot process instructions without a configured LLM.")
            return "LLM not configured. Please set it up in the Agent tab."

        # Construct a prompt for the LLM
        system_prompt = (
            "You are an expert AI assistant specializing in model optimization. "
            "Your task is to analyze user requests regarding a specific model file and determine the appropriate action. "
            "Available actions are: 'quantize', 'distill', 'analyze', 'none'. "
            "Respond with a single JSON object containing 'action' and 'parameters'. "
            "For 'quantize', parameters can be {'quant_type': '8-bit' | '4-bit' | 'FP16'}. "
            "For 'distill', you need a 'teacher_model_path'. "
            "For 'analyze', provide a 'description' of the model. "
            "If no action is appropriate, use 'none'."
        )
        full_prompt = f"{system_prompt}\\n\\nUser instruction: '{user_prompt}'\\nModel path: '{model_path}'\\n\\nJSON Response:"
        
        log_fn("INFO: Sending instructions to LLM for analysis...")
        response_text = llm_client.generate(full_prompt)
        log_fn(f"LLM Response: {response_text}")

        try:
            response_json = json.loads(response_text)
            action_from_llm = response_json.get("action", "none")
            params = response_json.get("parameters", {})

            if action_from_llm == "quantize":
                quant_type = params.get("quant_type", "8-bit") # Default
                log_fn(f"INFO: LLM decided to quantize with type: {quant_type}")
                # This needs a generic quantization function.
                # For now, we'll log it.
                return f"Agent decided to quantize the model to {quant_type}. Implementation pending."
            
            elif action_from_llm == "analyze":
                return params.get("description", "Analysis complete.")

            else: # none or other actions
                return "Agent acknowledged the instruction. No specific action taken."

        except json.JSONDecodeError:
            log_fn("Warning: LLM did not return valid JSON. Treating as a regular response.")
            return response_text

    # Original functionality: Execute a pre-defined action
    log_fn(f"INFO: Received action '{action}' for model at '{model_path}'.")
    try:
        if action == "Quantize to GGUF":
            log_fn("INFO: Starting GGUF quantization...")
            output_path = quantize_to_gguf(model_path, log_fn=log_fn)
            log_fn(f"SUCCESS: Quantized model saved to {output_path}")
            return f"Quantization successful. See logs for details."

        elif action == "Convert to ONNX":
            log_fn("INFO: Starting ONNX conversion from Diffusers pipeline...")
            output_path = convert_to_onnx_from_diffusers(model_path, log_fn=log_fn)
            log_fn(f"SUCCESS: ONNX model saved to {output_path}")
            return f"ONNX conversion successful. See logs for details."

        else:
            log_fn(f"WARNING: Action '{action}' is not yet implemented.")
            return f"Action '{action}' is not yet implemented."

    except Exception as e:
        import traceback
        error_msg = f"ERROR: Failed to execute action '{action}' on '{model_path}'. Reason: {e}"
        log_fn(error_msg)
        log_fn(traceback.format_exc())
        return error_msg
