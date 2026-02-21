"""
Full Gradio UI for the Distill & Quantize application with Knowledge Distillation and Local LLM support.
"""
import gradio as gr
import torch
import yaml
from pathlib import Path
import os
import pandas as pd
import subprocess
import tempfile
from agent import discover_models, analyze_artifacts, execute_agent_action
from core.logic import run_pipeline
from core.device_manager import get_device_manager
from core.local_llm import get_local_llm_client
from settings.app_settings import (
    MODEL_TYPES,
    QUANT_TYPES,
    DEFAULT_MODEL_TYPE,
    DEFAULT_QUANT_TYPE,
    GRADIO_TITLE,
    GRADIO_DESCRIPTION,
)

# Import compatibility patch for huggingface_hub
try:
    import huggingface_hub_compat
except ImportError:
    huggingface_hub_compat = None

# New import for BERT distillation and Quantization
from run_distillation import run_distillation_pipeline
from registry import STRATEGY_REGISTRY
from reference_guide import DISTILLATION_METHODS, QUANTIZATION_METHODS
from core.quantizer import quantize_model
from enhanced_file_browser import create_file_browser_ui
from convert_onnx import convert_onnx_fp16, convert_onnx_int8
from convert_pytorch import convert_pytorch_fp16, convert_pytorch_bf16
from convert_tensorrt import convert_tensorrt
from convert_openvino import convert_openvino
from synthetic_data_pipeline import run_synthetic_data_pipeline
from quantize_pytorch import quantize_pytorch_model
from distill_pytorch import run_pytorch_distillation, STRATEGY_MAP

def create_theme():
    """Create a custom dark theme with gradient background."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="radial-gradient(circle at 50% 0%, #1a1c2e 0%, #0f1016 100%)",
        block_background_fill="#1a1c2e",
        button_primary_background_fill="linear-gradient(90deg, #00f2fe 0%, #4facfe 100%)",
        button_primary_text_color="red",
        block_title_text_color="red",
        block_label_text_color="red",
        input_background_fill="rgba(30, 41, 59, 0.8)",
    )


class LogOutput:
    """Helper class to capture log output for Gradio."""
    def __init__(self):
        self.logs = []
    
    def log(self, msg: str):
        """Add a log message."""
        self.logs.append(msg)
        return "\n".join(self.logs)
    
    def clear(self):
        """Clear all logs."""
        self.logs = []
        return ""


def load_config():
    """Load settings from YAML config."""
    config_path = Path(__file__).parent / "config" / "settings.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def save_config(config: dict):
    """Save settings to YAML config."""
    config_path = Path(__file__).parent / "config" / "settings.yaml"
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def detect_local_llm():
    """Detect available local LLM providers."""
    from core.local_llm import get_local_llm_client
    client = get_local_llm_client()
    provider = client.detect_provider()
    models = client.get_available_models() if provider else []
    return provider or "None detected", ", ".join(models[:5]) if models else "No models found"


def get_local_llm_client_instance():
    """Get local LLM client instance."""
    from core.local_llm import get_local_llm_client as get_client
    config = load_config()
    llm_config = config.get("local_llm", {})
    return get_client(
        provider=llm_config.get("provider", "auto"),
        base_url=llm_config.get("base_url", "http://localhost:1234/v1"),
        ollama_url=llm_config.get("ollama_url", "http://localhost:11434"),
        api_key=llm_config.get("api_key", ""),
        model_name=llm_config.get("model_name", "")
    )


def process_model(
    model_path: str,
    model_type: str,
    quant_type: str,
    distillation_mode: str,
    teacher_model_path: str,
    teacher_model_type: str,
    teacher_model_arch: str,
    use_local_llm: bool,
    local_llm_provider: str,
    local_llm_url: str,
    smoothquant_alpha: float = 0.5,
    gptq_group_size: int = 128,
    progress=gr.Progress()
):
    """
    Process the model through distillation and quantization.
    
    Args:
        model_path: Path to the student model
        model_type: Type of model
        quant_type: Type of quantization
        distillation_mode: Distillation mode (placeholder or teacher_student)
        teacher_model_path: Path to teacher model (if teacher_student mode)
        teacher_model_type: Type of teacher model
        use_local_llm: Whether to use local LLM
        local_llm_provider: Local LLM provider
        local_llm_url: Local LLM URL
        progress: Gradio progress tracker
        
    Returns:
        Log output string
    """
    log_output = LogOutput()
    
    def log_fn(msg: str):
        log_output.log(msg)
        progress(0.5, desc=msg)
    
    try:
        if not model_path or not model_path.strip():
            return "ERROR: Please provide a model path."
        
        # Update config with UI settings
        config = load_config()
        if "distillation" not in config:
            config["distillation"] = {}
        
        config["distillation"]["mode"] = distillation_mode
        config["distillation"]["enabled"] = True
        
        if distillation_mode == "teacher_student":
            if not teacher_model_path or not teacher_model_path.strip():
                return "ERROR: Teacher model path is required for teacher-student distillation."
            config["distillation"]["teacher_model_path"] = teacher_model_path.strip()
            config["distillation"]["teacher_type"] = teacher_model_type
            config["distillation"]["teacher_arch"] = teacher_model_arch
        
        if "local_llm" not in config:
            config["local_llm"] = {}
        
        config["local_llm"]["enabled"] = use_local_llm
        if use_local_llm:
            config["local_llm"]["provider"] = local_llm_provider
            if local_llm_provider == "lm_studio" or local_llm_provider == "custom":
                config["local_llm"]["base_url"] = local_llm_url
            elif local_llm_provider == "ollama":
                config["local_llm"]["ollama_url"] = local_llm_url
        
        save_config(config)
        
        # Run the pipeline
        distilled_path, quantized_path = run_pipeline(
            model_path.strip(),
            model_type,
            quant_type,
            log_fn,
            smoothquant_alpha=smoothquant_alpha,
            gptq_group_size=gptq_group_size
        )
        
        log_output.log(f"\n✓ Distilled model saved to: {distilled_path}")
        log_output.log(f"✓ Quantized model saved to: {quantized_path}")
        log_output.log("\n=== SUCCESS ===")
        
        return log_output.log("")
    
    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        log_output.log(error_msg)
        import traceback
        log_output.log(traceback.format_exc())
        return log_output.log("")

def run_bert_distillation_process(
    teacher_path, student_path, custom_path, strategy, quant_type, 
    smoothquant_alpha=0.5, gptq_group_size=128, progress=gr.Progress()
):
    """Wrapper for running the BERT distillation pipeline from Gradio."""
    log_output = LogOutput()
    
    def log_fn(msg: str):
        log_output.log(msg)
        progress(0.5, desc=msg)

    try:
        if not teacher_path or not student_path:
            return "ERROR: Teacher and Student model paths are required."

        run_distillation_pipeline(
            teacher_model_path=teacher_path,
            student_model_path=student_path,
            custom_model_path=custom_path if custom_path else None,
            strategy_name=strategy,
            output_dir="bert_distilled_models",
            quantization_type=quant_type if quant_type != "None" else None,
            log_fn=log_fn,
            smoothquant_alpha=smoothquant_alpha,
            gptq_group_size=gptq_group_size
        )
        log_output.log("\n=== BERT Distillation SUCCESS ===")
        return log_output.log("")

    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        log_output.log(error_msg)
        import traceback
        log_output.log(traceback.format_exc())
        return log_output.log("")


def run_quantization_process(model_id, quant_level, progress=gr.Progress()):
    """Wrapper for running the quantization pipeline from Gradio."""
    log_output = LogOutput()
    
    def log_fn(msg: str):
        log_output.log(msg)
        progress(0.5, desc=msg)

    try:
        if not model_id:
            return "ERROR: Model ID or Path is required."

        zip_path = quantize_model(model_id, quant_level, log_fn=log_fn)
        log_output.log(f"\n✓ Quantized model zipped to: {zip_path}")
        log_output.log("\n=== QUANTIZATION SUCCESS ===")
        # Return the folder zip path as a file for Gradio
        return log_output.log(""), gr.update(value=zip_path, visible=True)

    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        log_output.log(error_msg)
        import traceback
        log_output.log(traceback.format_exc())
        return log_output.log(""), gr.update(visible=False)


def run_onnx_int8_conversion_process(model_path, progress=gr.Progress()):
    """Wrapper for running the ONNX INT8 conversion pipeline from Gradio."""
    log_output = LogOutput()

    def log_fn(msg: str):
        log_output.log(msg)
        progress(0.5, desc=msg)

    try:
        if not model_path:
            return "ERROR: Model Path is required."

        output_path = convert_onnx_int8(model_path, None, log_fn=log_fn)
        log_output.log(f"✓ Converted ONNX model saved to: {output_path}")
        log_output.log("=== ONNX INT8 Conversion SUCCESS ===")
        return log_output.log(""), gr.update(value=output_path, visible=True)

    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        log_output.log(error_msg)
        import traceback
        log_output.log(traceback.format_exc())
        return log_output.log(""), gr.update(visible=False)


def run_pytorch_bf16_conversion_process(model_path, progress=gr.Progress()):
    """Wrapper for running the PyTorch BF16 conversion pipeline from Gradio."""
    log_output = LogOutput()

    def log_fn(msg: str):
        log_output.log(msg)
        progress(0.5, desc=msg)

    try:
        if not model_path:
            return "ERROR: Model Path is required."

        output_path = convert_pytorch_bf16(model_path, None, log_fn=log_fn)
        log_output.log(f"✓ Converted PyTorch model saved to: {output_path}")
        log_output.log("=== PyTorch BF16 Conversion SUCCESS ===")
        return log_output.log(""), gr.update(value=output_path, visible=True)

    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        log_output.log(error_msg)
        import traceback
        log_output.log(traceback.format_exc())
        return log_output.log(""), gr.update(visible=False)


def run_onnx_conversion_process(model_path, progress=gr.Progress()):
    """Wrapper for running the ONNX FP16 conversion pipeline from Gradio."""
    log_output = LogOutput()

    def log_fn(msg: str):
        log_output.log(msg)
        progress(0.5, desc=msg)

    try:
        if not model_path:
            return "ERROR: Model Path is required."

        output_path = convert_onnx_fp16(model_path, None, log_fn=log_fn)
        log_output.log(f"✓ Converted ONNX model saved to: {output_path}")
        log_output.log("=== ONNX FP16 Conversion SUCCESS ===")
        return log_output.log(""), gr.update(value=output_path, visible=True)

    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        log_output.log(error_msg)
        import traceback
        log_output.log(traceback.format_exc())
        return log_output.log(""), gr.update(visible=False)


def run_pytorch_conversion_process(model_path, progress=gr.Progress()):
    """Wrapper for running the PyTorch FP16 conversion pipeline from Gradio."""
    log_output = LogOutput()

    def log_fn(msg: str):
        log_output.log(msg)
        progress(0.5, desc=msg)

    try:
        if not model_path:
            return "ERROR: Model Path is required."

        output_path = convert_pytorch_fp16(model_path, None, log_fn=log_fn)
        log_output.log(f"✓ Converted PyTorch model saved to: {output_path}")
        log_output.log("=== PyTorch FP16 Conversion SUCCESS ===")
        return log_output.log(""), gr.update(value=output_path, visible=True)

    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        log_output.log(error_msg)
        import traceback
        log_output.log(traceback.format_exc())
        return log_output.log(""), gr.update(visible=False)


def run_tensorrt_conversion_process(model_path, progress=gr.Progress()):
    """Wrapper for running the TensorRT conversion pipeline from Gradio."""
    log_output = LogOutput()

    def log_fn(msg: str):
        log_output.log(msg)
        progress(0.5, desc=msg)

    try:
        if not model_path:
            return "ERROR: Model Path is required."

        output_path = convert_tensorrt(model_path, None, log_fn=log_fn)
        log_output.log(f"✓ Converted TensorRT model saved to: {output_path}")
        log_output.log("=== TensorRT Conversion SUCCESS ===")
        return log_output.log(""), gr.update(value=output_path, visible=True)

    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        log_output.log(error_msg)
        import traceback
        log_output.log(traceback.format_exc())
        return log_output.log(""), gr.update(visible=False)


def run_openvino_conversion_process(model_path, progress=gr.Progress()):
    """Wrapper for running the OpenVINO conversion pipeline from Gradio."""
    log_output = LogOutput()

    def log_fn(msg: str):
        log_output.log(msg)
        progress(0.5, desc=msg)

    try:
        if not model_path:
            return "ERROR: Model Path is required."

        output_path = convert_openvino(model_path, None, log_fn=log_fn)
        log_output.log(f"✓ Converted OpenVINO model saved to: {output_path}")
        log_output.log("=== OpenVINO Conversion SUCCESS ===")
        return log_output.log(""), gr.update(value=output_path, visible=True)

    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        log_output.log(error_msg)
        import traceback
        log_output.log(traceback.format_exc())
        return log_output.log(""), gr.update(visible=False)


def run_synthetic_data_process(provider, api_key, base_url, model_name, num_records, output_dir, min_completeness, min_accuracy, progress=gr.Progress()):
    """Wrapper for running the synthetic data pipeline from Gradio."""
    log_output = LogOutput()

    def log_fn(msg: str):
        log_output.log(msg)
        progress(0.5, desc=msg)

    try:
        if provider == "openrouter":
            if not api_key or not api_key.strip():
                return "ERROR: OpenRouter API Key is required for OpenRouter provider."

            if not api_key.startswith("sk-or-v1-"):
                return "ERROR: Invalid OpenRouter API Key format. Should start with 'sk-or-v1-'"
        else:  # Local LLM providers
            if not base_url or not base_url.strip():
                return f"ERROR: Base URL is required for {provider} provider."

        dataset, output_path = run_synthetic_data_pipeline(
            api_key=api_key.strip() if api_key else None,
            provider=provider,
            base_url=base_url.strip() if base_url else None,
            model_name=model_name.strip() if model_name else None,
            num_records=int(num_records),
            output_dir=output_dir.strip() if output_dir else "synthetic_data_output",
            min_completeness=min_completeness,
            min_accuracy=min_accuracy,
            log_fn=log_fn
        )
        
        if dataset is not None and output_path:
            log_output.log(f"\n✓ Synthetic dataset generated successfully!")
            log_output.log(f"✓ Dataset saved to: {output_path}")
            log_output.log(f"✓ Total records generated: {len(dataset)}")
            log_output.log("\n=== SYNTHETIC DATA PIPELINE SUCCESS ===")
            return log_output.log(""), gr.update(value=output_path, visible=True)
        else:
            return log_output.log("Failed to generate synthetic dataset"), gr.update(visible=False)

    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        log_output.log(error_msg)
        import traceback
        log_output.log(traceback.format_exc())
        return log_output.log(""), gr.update(visible=False)


def get_device_info():
    """Get current device information for display."""
    device_manager = get_device_manager()
    device_name = device_manager.get_device_name()
    
    info_lines = [f"**Device:** {device_name}"]
    
    if torch.cuda.is_available():
        vram_info = device_manager.get_vram_info()
        if vram_info:
            total, allocated, free = vram_info
            info_lines.append(f"**VRAM:** {allocated:.2f}GB / {total:.2f}GB (Free: {free:.2f}GB)")
    
    info_lines.append("\n*The app will automatically use CPU if VRAM is limited.*")
    
    return "\n".join(info_lines)


custom_theme = create_theme()

css = """
    body { background: radial-gradient(circle at 50% 0%, #1a1c2e 0%, #0f1016 100%) !important; }
    .gradio-container { background: transparent !important; }
    h1 { font-size: 3.5rem !important; text-align: center !important; margin-bottom: 1rem !important; }
    
    /* GLOBAL TEXT RED (Nuclear Option) - Expanded to strong/em/buttons */
    .gradio-container, .prose, h1, h2, h3, h4, h5, p, label, .block-label, span, div, li, td, strong, em, b, i, th { 
        color: red !important; 
    }
    
    /* --- INPUTS & TEXTAREAS --- */
    input, textarea { 
        background-color: white !important; 
        color: red !important; 
        border: 1px solid red !important;
    }
    input::placeholder, textarea::placeholder { color: rgba(255, 0, 0, 0.5) !important; }
    
    /* --- DROPDOWNS & SELECTS --- */
    /* Container for the selected value and the options list */
    .wrap-inner, .dropdown-wrapper, .options, .single-select, .token, .item {
         background-color: white !important;
         color: red !important;
         border-color: red !important;
    }
    
    /* Text inside dropdowns (selected value, options list items) */
    .wrap-inner span, .dropdown-wrapper span, .options span, .item span, 
    .single-select, .token span, li.item {
        color: red !important;
    }
    
    /* Fix for SVG icons (arrows/chevrons) in dropdowns */
    .dropdown-wrapper svg, .wrap-inner svg {
        fill: red !important;
        stroke: red !important;
    }

    /* --- CHECKBOXES & RADIOS --- */
    .checkbox label span, .radio label span {
        color: red !important;
    }
    
    /* Buttons */
    button, .lg.primary, .lg.secondary, .sm.primary, .sm.secondary, .primary { color: red !important; } 

    /* --- TABS SCROLLING --- */
    .tab-buttons {
        overflow-x: auto !important;
        white-space: nowrap !important;
        flex-wrap: nowrap !important;
    }
    """

def create_ui():
    """Create and return the Gradio interface."""
    
    # Define default paths for browser
    TEACHER_ROOT = os.getcwd()
    API_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    
    # Load initial config
    config = load_config()
    dist_config = config.get("distillation", {})
    llm_config = config.get("local_llm", {})
    
    with gr.Blocks() as demo:
        gr.Markdown(f"# {GRADIO_TITLE}")
        gr.Markdown(GRADIO_DESCRIPTION)
        
        with gr.Tabs():
            with gr.TabItem("Diffusion Distillation"):
                # Device information panel
                with gr.Row():
                    device_info = gr.Markdown(get_device_info())
                
                # 1. Input Model Selection Section
                gr.Markdown("## Input Model Selection (Random/Student Model)")
                with gr.Row():
                    with gr.Column(scale=3):
                        model_path = gr.Textbox(
                            label="Input Model Path (Target for Processing)",
                            placeholder="Select a folder path from the Browser tab...",
                            info="Path to the model you want to quantize/distill (e.g. HuggingFace folder)"
                        )
                    
                    with gr.Column(scale=2):
                        model_type = gr.Dropdown(
                            choices=MODEL_TYPES,
                            value=DEFAULT_MODEL_TYPE,
                            label="Input Model Library/Type",
                            info="Select the library or type of the model"
                        )
                
                # 2. Knowledge Distillation Configuration Section
                gr.Markdown("## Knowledge Distillation Configuration")
                distillation_mode = gr.Radio(
                    choices=["placeholder", "teacher_student"],
                    value=dist_config.get("mode", "placeholder"),
                    label="Distillation Mode",
                    info="Placeholder: No training | Teacher-Student: Real KD training"
                )
                
                with gr.Row(visible=(dist_config.get("mode", "placeholder") == "teacher_student")) as teacher_row:
                    with gr.Column(scale=3):
                        teacher_model_path = gr.Textbox(
                            label="Teacher Model Path",
                            placeholder="Select a folder path from the Browser tab...",
                            value=dist_config.get("teacher_model_path", ""),
                            info="Required for teacher-student mode"
                        )

                        teacher_model_type = gr.Dropdown(
                            choices=MODEL_TYPES,
                            value=dist_config.get("teacher_type", DEFAULT_MODEL_TYPE),
                            label="Teacher Model Source/Type"
                        )
                    
                    with gr.Column(scale=2):
                        teacher_model_arch = gr.Dropdown(
                            choices=["Causal LM (GPT, Llama)", "Masked LM (BERT, RoBERTa)", "Generic AutoModel"],
                            value=dist_config.get("teacher_arch", "Causal LM (GPT, Llama)"),
                            label="Teacher Architecture",
                            info="Select based on teacher model type"
                        )

                def toggle_teacher_visibility(mode):
                    return gr.update(visible=(mode == "teacher_student"))
                
                distillation_mode.change(fn=toggle_teacher_visibility, inputs=[distillation_mode], outputs=[teacher_row])
                
                # 3. Local LLM Configuration Section
                gr.Markdown("## Local LLM Integration (Optional)")
                use_local_llm = gr.Checkbox(
                    label="Use Local LLM",
                    value=llm_config.get("enabled", False),
                    info="Connect to LM Studio, Ollama, or other local LLM servers"
                )
                
                with gr.Row(visible=llm_config.get("enabled", False)) as llm_row:
                    local_llm_provider = gr.Dropdown(
                        choices=["auto", "lm_studio", "ollama", "custom"],
                        value=llm_config.get("provider", "auto"),
                        label="LLM Provider"
                    )
                    local_llm_url = gr.Textbox(
                        label="LLM Server URL",
                        value=llm_config.get("base_url", "http://localhost:1234/v1"),
                        info="Base URL for LLM server"
                    )
                
                use_local_llm.change(fn=lambda x: gr.update(visible=x), inputs=[use_local_llm], outputs=[llm_row])
                
                # 4. Quantization Configuration Section
                gr.Markdown("## Quantization Configuration")
                quant_type = gr.Dropdown(
                    choices=QUANT_TYPES,
                    value=DEFAULT_QUANT_TYPE,
                    label="Quantization Type",
                )
                
                with gr.Row(visible=False) as sq_params:
                    sq_alpha = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="SmoothQuant Alpha")
                
                with gr.Row(visible=False) as gptq_params:
                    gptq_gs = gr.Dropdown(choices=[32, 64, 128, 256], value=128, label="GPTQ Group Size")

                def update_quant_params(qtype):
                    return gr.update(visible=qtype == "SmoothQuant (INT4)"), gr.update(visible=qtype == "GPTQ (4-bit)")
                
                quant_type.change(fn=update_quant_params, inputs=[quant_type], outputs=[sq_params, gptq_params])
                
                run_btn = gr.Button("Run Distill + Quantize", variant="primary")
                log_output = gr.Textbox(label="Live Log Output", lines=20, interactive=False)
                
                run_btn.click(
                    fn=process_model,
                    inputs=[
                        model_path, model_type, quant_type, distillation_mode,
                        teacher_model_path, teacher_model_type, teacher_model_arch,
                        use_local_llm, local_llm_provider, local_llm_url, 
                        sq_alpha, gptq_gs
                    ],
                    outputs=[log_output]
                ).then(fn=lambda: get_device_info(), outputs=[device_info])

            with gr.TabItem("BERT Distillation"):
                gr.Markdown("## BERT Distillation and Quantization")
                
                with gr.Row():
                    bert_teacher_path = gr.Textbox(label="Teacher Model", placeholder="Select path from Browser tab...")
                
                with gr.Row():
                    bert_student_path = gr.Textbox(label="Student Model", placeholder="Select path from Browser tab...")

                with gr.Row():
                    bert_custom_path = gr.Textbox(label="Custom Model (Optional)", placeholder="Select path from Browser tab...")
                
                bert_strategy = gr.Dropdown(
                    choices=list(STRATEGY_REGISTRY.keys()),
                    label="Distillation Strategy",
                    value="patient_kd"
                )
                
                bert_quant_type = gr.Dropdown(
                    choices=["None"] + QUANT_TYPES,
                    label="Quantization Type",
                    value="None"
                )
                
                bert_run_btn = gr.Button("Run BERT Distillation", variant="primary")
                bert_log_output = gr.Textbox(label="Live Log Output", lines=20, interactive=False)

                bert_run_btn.click(
                    fn=run_bert_distillation_process,
                    inputs=[bert_teacher_path, bert_student_path, bert_custom_path, bert_strategy, bert_quant_type, sq_alpha, gptq_gs],
                    outputs=[bert_log_output]
                )

            with gr.TabItem("PyTorch Distillation"):
                gr.Markdown("## PyTorch Model Distillation")
                gr.Markdown(
                    "Distill knowledge from a teacher model to a student model using various strategies."
                )
                with gr.Row():
                    pt_distill_teacher_path = gr.Textbox(label="Teacher Model Path", placeholder="Select path from Browser tab...")
                with gr.Row():
                    pt_distill_student_path = gr.Textbox(label="Student Model Path", placeholder="Select path from Browser tab...")
                with gr.Row():
                    pt_distill_custom_path = gr.Textbox(label="Custom Model Path (for MultiTeacherKD)", placeholder="Select path from Browser tab...")
                with gr.Row():
                    pt_distill_data_path = gr.File(label="Training Data (.txt file with sentences)", file_count="single", file_types=[".txt"])

                pt_distill_strategy = gr.Dropdown(
                    choices=list(STRATEGY_MAP.keys()),
                    label="Distillation Strategy",
                    value="LogitKD"
                )
                
                pt_distill_run_btn = gr.Button("Run PyTorch Distillation", variant="primary")
                pt_distill_log_output = gr.Textbox(label="Live Log Output", lines=20, interactive=False)

                def run_pytorch_distill_process(teacher_path, student_path, custom_path, data_path, strategy, progress=gr.Progress()):
                    log_output = LogOutput()
                    def log_fn(msg: str):
                        log_output.log(msg)
                        progress(0.5, desc=msg)
                    
                    try:
                        if not teacher_path or not student_path or not data_path:
                            return "ERROR: Teacher, Student, and Data paths are required."

                        output_dir = "pytorch_distilled_models"
                        distilled_path = run_pytorch_distillation(
                            teacher_path,
                            student_path,
                            data_path.name,
                            strategy,
                            output_dir,
                            custom_path if custom_path else None,
                            log_fn=log_fn
                        )
                        log_output.log(f"\\n=== PyTorch Distillation SUCCESS ===")
                        log_output.log(f"Distilled model saved to: {distilled_path}")
                        return log_output.log("")

                    except Exception as e:
                        error_msg = f"ERROR: {str(e)}"
                        log_output.log(error_msg)
                        import traceback
                        log_output.log(traceback.format_exc())
                        return log_output.log("")

                pt_distill_run_btn.click(
                    fn=run_pytorch_distill_process,
                    inputs=[pt_distill_teacher_path, pt_distill_student_path, pt_distill_custom_path, pt_distill_data_path, pt_distill_strategy],
                    outputs=[pt_distill_log_output]
                )

            with gr.TabItem("FP16 Conversion"):
                gr.Markdown("## Model FP16 (Half-Precision) Conversion")
                gr.Markdown(
                    "Convert your ONNX and PyTorch models to FP16 (half-precision) to reduce file size and improve inference speed on supported hardware (like NVIDIA GPUs)."
                )
                with gr.Tabs():
                    with gr.TabItem("ONNX to FP16"):
                        with gr.Row():
                            onnx_model_path_in = gr.Textbox(label="Input ONNX Model Path", placeholder="Select path from Browser tab...")
                        
                        onnx_quant_type = gr.Dropdown(choices=["FP16", "INT8"], label="ONNX Conversion Type", value="FP16")
                        onnx_convert_btn = gr.Button("Convert ONNX", variant="primary")
                        onnx_download = gr.File(label="Download Converted ONNX Model", visible=False)
                        onnx_log_output = gr.Textbox(label="ONNX Conversion Log", lines=15, interactive=False)

                        def onnx_conversion_dispatcher(model_path, quant_type):
                            if quant_type == "INT8":
                                return run_onnx_int8_conversion_process(model_path)
                            return run_onnx_conversion_process(model_path)
                        
                        onnx_convert_btn.click(
                            fn=onnx_conversion_dispatcher,
                            inputs=[onnx_model_path_in, onnx_quant_type],
                            outputs=[onnx_log_output, onnx_download]
                        )

                    with gr.TabItem("PyTorch Conversion"):
                        with gr.Row():
                            pytorch_model_path_in = gr.Textbox(label="Input PyTorch Model Path", placeholder="Select path from Browser tab...")

                        pytorch_quant_type = gr.Dropdown(choices=["FP16", "BF16"], label="PyTorch Conversion Type", value="FP16")
                        pytorch_convert_btn = gr.Button("Convert PyTorch", variant="primary")
                        pytorch_download = gr.File(label="Download Converted PyTorch Model", visible=False)
                        pytorch_log_output = gr.Textbox(label="PyTorch Conversion Log", lines=15, interactive=False)
                        
                        def pytorch_conversion_dispatcher(model_path, quant_type):
                            if quant_type == "BF16":
                                return run_pytorch_bf16_conversion_process(model_path)
                            return run_pytorch_conversion_process(model_path)

                        pytorch_convert_btn.click(
                            fn=pytorch_conversion_dispatcher,
                            inputs=[pytorch_model_path_in, pytorch_quant_type],
                            outputs=[pytorch_log_output, pytorch_download]
                        )

            with gr.TabItem("PyTorch PT2E Quantization"):
                gr.Markdown("## PyTorch 2.x Post-Training Quantization")
                gr.Markdown(
                    "Quantize PyTorch models using the `torch.export` workflow with `torchao`."
                    "This requires a full model file (`.pth`) and optional calibration data."
                )
                with gr.Row():
                    pt_quant_model_path_in = gr.Textbox(label="Input PyTorch Model Path (.pth)", placeholder="Select a .pth model file...")
                with gr.Row():
                    pt_quant_calibration_in = gr.File(label="Calibration Data (.pt tensor file)", file_count="single", file_types=[".pt"])
                
                pt_quant_convert_btn = gr.Button("Quantize PyTorch Model", variant="primary")
                pt_quant_download = gr.File(label="Download Quantized PyTorch Model", visible=False)
                pt_quant_log_output = gr.Textbox(label="Quantization Log", lines=15, interactive=False)

                def run_pytorch_quant_process(model_path, calib_path, progress=gr.Progress()):
                    log_output = LogOutput()
                    def log_fn(msg: str):
                        log_output.log(msg)
                        progress(0.5, desc=msg)
                    
                    try:
                        if not model_path:
                            return "ERROR: Model Path is required.", gr.update(visible=False)

                        # Determine the python executable from the venv
                        python_executable = os.path.join(os.path.dirname(__file__), 'env', 'Scripts', 'python.exe')
                        if not os.path.exists(python_executable):
                           python_executable = os.path.join(os.path.dirname(__file__), 'env', 'bin', 'python')

                        if not os.path.exists(python_executable):
                           log_fn("ERROR: Could not find python executable in virtual environment.")
                           return log_output.log(""), gr.update(visible=False)

                        quant_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'quantize_pytorch.py'))
                        
                        cmd = [
                            python_executable,
                            quant_script_path,
                            "--model_path", model_path,
                        ]
                        if calib_path:
                            cmd.extend(["--calibration_data_path", calib_path.name])

                        # Create a temporary file to store the output path
                        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".txt") as tmpfile:
                            output_path_file = tmpfile.name
                        
                        cmd.extend(["--output_path_file", output_path_file])
                        
                        log_fn(f"Running command: {' '.join(cmd)}")
                        
                        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
                        
                        for line in process.stdout:
                            log_fn(line.strip())

                        process.wait()

                        with open(output_path_file, 'r') as f:
                            output_path = f.read().strip()
                        
                        os.remove(output_path_file)

                        if process.returncode == 0 and output_path:
                            log_output.log(f"✓ Quantized PyTorch model saved to: {output_path}")
                            log_output.log("=== PyTorch Quantization SUCCESS ===")
                            return log_output.log(""), gr.update(value=output_path, visible=True)
                        else:
                            log_output.log("=== PyTorch Quantization FAILED ===")
                            return log_output.log(""), gr.update(visible=False)

                    except Exception as e:
                        error_msg = f"ERROR: {str(e)}"
                        log_output.log(error_msg)
                        import traceback
                        log_output.log(traceback.format_exc())
                        return log_output.log(""), gr.update(visible=False)

                pt_quant_convert_btn.click(
                    fn=run_pytorch_quant_process,
                    inputs=[pt_quant_model_path_in, pt_quant_calibration_in],
                    outputs=[pt_quant_log_output, pt_quant_download]
                )

            with gr.TabItem("Hardware-Specific Conversion"):
                gr.Markdown("## Hardware-Specific Model Conversions")
                gr.Markdown(
                    "Optimize your models for specific hardware targets like NVIDIA GPUs (TensorRT) or Intel CPUs/iGPUs (OpenVINO) for maximum performance."
                )
                with gr.Tabs():
                    with gr.TabItem("NVIDIA TensorRT"):
                        with gr.Row():
                            tensorrt_model_path_in = gr.Textbox(label="Input PyTorch Model Path (.pth)", placeholder="Select path from Browser tab...")
                        
                        tensorrt_convert_btn = gr.Button("Convert to TensorRT", variant="primary")
                        tensorrt_download = gr.File(label="Download Converted TensorRT Model", visible=False)
                        tensorrt_log_output = gr.Textbox(label="TensorRT Conversion Log", lines=15, interactive=False)

                        tensorrt_convert_btn.click(
                            fn=run_tensorrt_conversion_process,
                            inputs=[tensorrt_model_path_in],
                            outputs=[tensorrt_log_output, tensorrt_download]
                        )

                    with gr.TabItem("Intel OpenVINO"):
                        with gr.Row():
                            openvino_model_path_in = gr.Textbox(label="Input Model Path (ONNX, PyTorch, etc.)", placeholder="Select path from Browser tab...")

                        openvino_convert_btn = gr.Button("Convert to OpenVINO", variant="primary")
                        openvino_download = gr.File(label="Download Converted OpenVINO Model", visible=False)
                        openvino_log_output = gr.Textbox(label="OpenVINO Conversion Log", lines=15, interactive=False)

                        openvino_convert_btn.click(
                            fn=run_openvino_conversion_process,
                            inputs=[openvino_model_path_in],
                            outputs=[openvino_log_output, openvino_download]
                        )

            with gr.TabItem("Quantize Any Model"):
                gr.Markdown("## 🌌 AI Model Shrinker: Quantize Your Models!")
                gr.Markdown(
                    "Enter a Hugging Face Model ID to effortlessly quantize it and reduce its size and memory footprint. "
                    "This can significantly improve inference speed and allow larger models to run on more modest hardware. "
                    "<br><b>Important Notes:</b>"
                    "<ul>"
                    "<li><b>GPU Required:</b> 8-bit and 4-bit quantization (using `bitsandbytes`) require a **CUDA-enabled GPU** to work properly.</li>"
                    "<li><b>Compatibility:</b> Not all models are guaranteed to work perfectly after quantization, especially 4-bit.</li>"
                    "<li><b>Downloading:</b> The output will be a `.zip` file containing the quantized model's directory.</li>"
                    "</ul>"
                )
                
                with gr.Row():
                    with gr.Column(scale=3):
                        quant_model_id = gr.Textbox(
                            label="Hugging Face Model ID or Local Path",
                            placeholder="e.g., stabilityai/stablelm-zephyr-3b or meta-llama/Llama-2-7b-hf"
                        )

                quant_level_choice = gr.Dropdown(
                    choices=["8-bit (INT8)", "4-bit (INT4)", "FP16 (Half-Precision)"],
                    label="Select Quantization Level",
                    value="8-bit (INT8)"
                )
                
                quant_run_btn = gr.Button("🚀 Start Quantization", variant="primary")
                
                quant_download = gr.File(label="Download Quantized Model", visible=False)
                quant_log_output = gr.Textbox(label="Quantization Log", lines=15, interactive=False)

                quant_run_btn.click(
                    fn=run_quantization_process,
                    inputs=[quant_model_id, quant_level_choice],
                    outputs=[quant_log_output, quant_download]
                )

            with gr.TabItem("🧪 Synthetic Data Pipeline"):
                gr.Markdown("## 🧪 Synthetic Data & Distillation Pipeline")
                gr.Markdown(
                    "Generate high-quality, domain-specific synthetic datasets using NVIDIA NeMo Data Designer "
                    "and OpenRouter distillable endpoints. Create license-safe datasets ready for model distillation "
                    "and fine-tuning workflows."
                )
                
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### � LLM Provider Selection")
                        llm_provider = gr.Dropdown(
                            choices=["openrouter", "ollama", "lm_studio", "custom"],
                            value="openrouter",
                            label="LLM Provider",
                            info="Choose between OpenRouter (cloud) or local LLM servers"
                        )
                        
                        with gr.Group(visible=True) as openrouter_group:
                            openrouter_api_key = gr.Textbox(
                                label="OpenRouter API Key",
                                placeholder="sk-or-v1-...",
                                type="password",
                                info="Get your API key from https://openrouter.ai/keys"
                            )
                        
                        with gr.Group(visible=False) as local_llm_group:
                            local_base_url = gr.Textbox(
                                label="Local LLM Server URL",
                                placeholder="http://localhost:11434",
                                info="URL for your local LLM server"
                            )
                            
                            local_model_name = gr.Textbox(
                                label="Model Name (Optional)",
                                placeholder="Leave empty to auto-detect",
                                info="Specific model to use, or leave empty for first available"
                            )
                        
                        gr.Markdown("### 📊 Dataset Configuration")
                        num_records = gr.Slider(
                            minimum=10,
                            maximum=1000,
                            value=100,
                            step=10,
                            label="Number of Records to Generate",
                            info="More records = longer processing time"
                        )
                        
                        output_directory = gr.Textbox(
                            label="Output Directory",
                            value="synthetic_data_output",
                            placeholder="Directory name for generated datasets",
                            info="Where to save generated CSV files"
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 🎯 Quality Filtering")
                        min_completeness = gr.Dropdown(
                            choices=["Incomplete", "PartiallyComplete", "Complete"],
                            value="PartiallyComplete",
                            label="Minimum Completeness Score",
                            info="Filter for answer completeness"
                        )
                        
                        min_accuracy = gr.Dropdown(
                            choices=["Inaccurate", "PartiallyAccurate", "Accurate"],
                            value="PartiallyAccurate",
                            label="Minimum Accuracy Score",
                            info="Filter for factual accuracy"
                        )
                        
                        gr.Markdown("### ℹ️ Provider Information")
                        gr.Markdown(
                            """
                            **OpenRouter**: Cloud-based, license-safe distillable endpoints
                            - Cost: ~$0.10 per 100 records
                            - Requires API key
                            - High quality models (Nemotron 3 Nano)
                            
                            **Ollama**: Local LLM server
                            - URL: http://localhost:11434
                            - Free (local hardware required)
                            - Models: Llama2, Mistral, CodeLlama, etc.
                            
                            **LM Studio**: Local LLM server
                            - URL: http://localhost:1234/v1
                            - Free (local hardware required)
                            - OpenAI-compatible API
                            
                            **Custom**: Any OpenAI-compatible server
                            - Configure URL as needed
                            - Works with most local LLM setups
                            """
                        )
                
                def update_provider_visibility(provider):
                    if provider == "openrouter":
                        return gr.update(visible=True), gr.update(visible=False)
                    else:
                        return gr.update(visible=False), gr.update(visible=True)
                
                llm_provider.change(
                    fn=update_provider_visibility,
                    inputs=[llm_provider],
                    outputs=[openrouter_group, local_llm_group]
                )
                
                synthetic_run_btn = gr.Button("🚀 Generate Synthetic Dataset", variant="primary")
                
                synthetic_download = gr.File(label="Download Generated Dataset", visible=False)
                synthetic_log_output = gr.Textbox(label="Pipeline Log", lines=20, interactive=False)
                
                synthetic_run_btn.click(
                    fn=run_synthetic_data_process,
                    inputs=[llm_provider, openrouter_api_key, local_base_url, local_model_name, num_records, output_directory, min_completeness, min_accuracy],
                    outputs=[synthetic_log_output, synthetic_download]
                )

            with gr.TabItem("🤖 Agent"):
                gr.Markdown("## 🤖 AI Agent: Project Inspector")
                gr.Markdown("An intelligent agent to scan, analyze, and manage your local AI models and project artifacts.")

                with gr.Accordion("LLM Configuration", open=True):
                    agent_llm_config = config.get("agent_llm", {})
                    
                    agent_llm_provider = gr.Dropdown(
                        choices=[
                            "None",
                            # Local & Open Source Frameworks
                            "LM Studio (Local)",
                            "Ollama (Local)",
                            "LangChain",
                            "LlamaIndex",
                            "Open Interpreter",
                            "Lobe Chat",
                            "Jan",
                            "FastChat",
                            # Open Source Agent Frameworks
                            "openclaw",
                            "OpenHands",
                            "AutoGen",
                            "Flowise",
                            "OpenDevin",
                            "MetaGPT",
                            "PrivateGPT",
                            "cal.ai",
                            "mem0",
                            "AgentGPT",
                            "Vibe Kanban",
                            "SuperAGI",
                            "ChatDev",
                            "Camel",
                            "OpenAGI",
                            # --- Closed Source Models ---
                            # OpenAI
                            "OpenAI: GPT-5.2",
                            "OpenAI: GPT-5.1",
                            "OpenAI: GPT-4o",
                            "OpenAI: GPT-4.1",
                            "OpenAI: GPT-4 Turbo",
                            "OpenAI: GPT-4",
                            "OpenAI: GPT-3.5 Series",
                            # Anthropic
                            "Anthropic: Claude 3 Opus",
                            "Anthropic: Claude 3 Sonnet",
                            "Anthropic: Claude 3 Haiku",
                            "Anthropic: Claude 2.1",
                            "Anthropic: Claude 2.0",
                            # Google
                            "Google: Gemini Ultra",
                            "Google: Gemini Pro",
                            "Google: Gemini Flash",
                            "Google: PaLM 2",
                            "Google: LaMDA",
                            # Microsoft
                            "Microsoft: Phi-3",
                            # Cohere
                            "Cohere: Command R+",
                            "Cohere: Command X",
                            "Cohere: Command Light",
                            # Other Major Models
                            "Amazon: Titan Text",
                            "Mistral: Large (API)",
                            "Inflection: Pi (Inflection-2)",
                            "xAI: Grok-2 (API)",
                            "xAI: Grok-1.5 (API)",
                            "Baidu: ERNIE Bot",
                            "Alibaba: Qwen-VL Max (API)",
                            "Tencent: Hunyuan",
                            "Huawei: Pangu",
                        ],
                        label="LLM Provider for Agent Reasoning",
                        value=agent_llm_config.get("provider", "None"),
                        info="Select the LLM the agent will use for analysis."
                    )

                    agent_api_key = gr.Textbox(
                        label="API Key", 
                        placeholder="Enter your API key...",
                        value=agent_llm_config.get("api_key", ""),
                        visible=(agent_llm_config.get("provider", "None") in ["OpenAI", "Google Gemini"])
                    )
                    
                    # Determine initial visibility and value for endpoint URL
                    initial_provider = agent_llm_config.get("provider", "None")
                    initial_endpoint_visible = initial_provider in ["Ollama (Local)", "LM Studio (Local)"]
                    initial_endpoint_value = agent_llm_config.get("endpoint_url", "")
                    if not initial_endpoint_value:
                        if initial_provider == "Ollama (Local)":
                            initial_endpoint_value = "http://localhost:11434"
                        elif initial_provider == "LM Studio (Local)":
                             initial_endpoint_value = "http://localhost:1234/v1"

                    agent_endpoint_url = gr.Textbox(
                        label="Endpoint URL", 
                        placeholder="e.g., http://localhost:11434",
                        value=initial_endpoint_value,
                        visible=initial_endpoint_visible
                    )

                    save_agent_config_button = gr.Button("Save LLM Configuration")
                    config_save_status = gr.Markdown()

                    def update_provider_fields(provider):
                        is_cloud = provider in ["OpenAI", "Google Gemini"]
                        is_local = provider in ["Ollama (Local)", "LM Studio (Local)"]
                        
                        endpoint_update_params = {"visible": is_local}
                        if provider == "Ollama (Local)":
                            endpoint_update_params["placeholder"] = "http://localhost:11434"
                            endpoint_update_params["value"] = "http://localhost:11434"
                        elif provider == "LM Studio (Local)":
                            endpoint_update_params["placeholder"] = "http://localhost:1234/v1"
                            endpoint_update_params["value"] = "http://localhost:1234/v1"

                        return gr.update(visible=is_cloud), gr.update(**endpoint_update_params)

                    agent_llm_provider.change(
                        fn=update_provider_fields,
                        inputs=[agent_llm_provider],
                        outputs=[agent_api_key, agent_endpoint_url]
                    )

                    def save_agent_config(provider, api_key, endpoint_url):
                        current_config = load_config()
                        current_config["agent_llm"] = {
                            "provider": provider,
                            "api_key": api_key,
                            "endpoint_url": endpoint_url
                        }
                        save_config(current_config)
                        return "✅ Configuration saved successfully."

                    save_agent_config_button.click(
                        fn=save_agent_config,
                        inputs=[agent_llm_provider, agent_api_key, agent_endpoint_url],
                        outputs=[config_save_status]
                    )

                with gr.Row():
                    agent_scan_button = gr.Button("Scan Project for Models", variant="secondary")
                
                agent_results_df = gr.DataFrame(
                    lambda: pd.DataFrame(columns=['Type', 'Path', 'Details', 'Suggested Action']),
                    label="Discovered Model Artifacts",
                    interactive=True
                )

                with gr.Row():
                    agent_file_selection = gr.Dropdown(label="Select a file to process", interactive=True)
                    agent_instruction_box = gr.Textbox(label="Instructions for the Agent", placeholder="e.g., 'Quantize this model to 8-bit' or 'Tell me more about this file.'")

                run_agent_button = gr.Button("Run Agent with Instructions", variant="primary")
                agent_log_output = gr.Textbox(label="Agent Log", lines=15, interactive=False)
                
                # Store the full dataframe in a hidden state for access
                df_state = gr.State()

                def handle_agent_scan_and_analyze():
                    project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
                    discovered_df = discover_models(project_root)
                    analyzed_df = analyze_artifacts(discovered_df)
                    # Update the dropdown choices
                    file_choices = analyzed_df["Path"].tolist()
                    return analyzed_df, analyzed_df, gr.update(choices=file_choices)

                agent_scan_button.click(
                    fn=handle_agent_scan_and_analyze,
                    inputs=[],
                    outputs=[agent_results_df, df_state, agent_file_selection]
                )

                def handle_run_agent_instruction(selected_file, instructions):
                    log_output = LogOutput()
                    def log_fn(msg):
                        log_output.log(msg)
                    
                    if not selected_file or not instructions:
                        return "ERROR: Please select a file and provide instructions."

                    project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
                    absolute_model_path = os.path.join(project_root, selected_file)

                    # This will now be a more generic function
                    result = execute_agent_action(
                        model_path=absolute_model_path, 
                        action=None, # Action is now determined by LLM
                        log_fn=log_fn,
                        user_prompt=instructions
                    )
                    log_fn(f"Agent execution finished.")
                    return log_output.log("")

                run_agent_button.click(
                    fn=handle_run_agent_instruction,
                    inputs=[agent_file_selection, agent_instruction_box],
                    outputs=[agent_log_output]
                )

            with gr.TabItem("📂 Browser"):
                browser_ui, browser_selected_path = create_file_browser_ui()
                with gr.Row():
                    copy_to_student = gr.Button("Use for Student Model")
                    copy_to_teacher = gr.Button("Use for Teacher Model")
                    copy_to_bert_student = gr.Button("Use for BERT Student")
                    copy_to_bert_teacher = gr.Button("Use for BERT Teacher")
                    copy_to_quantize_input = gr.Button("Use for Quantize Input")
                    copy_to_onnx_fp16_input = gr.Button("Use for ONNX FP16 Input")
                    copy_to_pytorch_fp16_input = gr.Button("Use for PyTorch FP16 Input")
                    copy_to_tensorrt_input = gr.Button("Use for TensorRT Input")
                    copy_to_openvino_input = gr.Button("Use for OpenVINO Input")
                
                # Connect the browser's selected_path to the main UI textboxes
                copy_to_student.click(lambda x: x, inputs=browser_selected_path, outputs=model_path)
                copy_to_teacher.click(lambda x: x, inputs=browser_selected_path, outputs=teacher_model_path)
                copy_to_bert_student.click(lambda x: x, inputs=browser_selected_path, outputs=bert_student_path)
                copy_to_bert_teacher.click(lambda x: x, inputs=browser_selected_path, outputs=bert_teacher_path)
                copy_to_quantize_input.click(lambda x: x, inputs=browser_selected_path, outputs=quant_model_id)
                copy_to_onnx_fp16_input.click(lambda x: x, inputs=browser_selected_path, outputs=onnx_model_path_in)
                copy_to_pytorch_fp16_input.click(lambda x: x, inputs=browser_selected_path, outputs=pytorch_model_path_in)
                copy_to_tensorrt_input.click(lambda x: x, inputs=browser_selected_path, outputs=tensorrt_model_path_in)
                copy_to_openvino_input.click(lambda x: x, inputs=browser_selected_path, outputs=openvino_model_path_in)

            with gr.TabItem("Reference Guide"):
                gr.Markdown("## Comprehensive Method Maps")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Knowledge Distillation Methods")
                        gr.TextArea(value=DISTILLATION_METHODS, show_label=False, lines=30, interactive=False)
                    with gr.Column():
                        gr.Markdown("### Quantization Formats")
                        gr.TextArea(value=QUANTIZATION_METHODS, show_label=False, lines=30, interactive=False)

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=custom_theme, css=css)
