
import torch
import os
import sys

# This ensures the script can find packages installed in the virtual environment.
venv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'env', 'Lib', 'site-packages'))
if venv_path not in sys.path:
    sys.path.insert(0, venv_path)

from torch.utils.data import TensorDataset, DataLoader

# Conditional import for torchao, handle if not installed
try:
    from torchao.quantization.pt2e import prepare_pt2e, convert_pt2e
    from executorch.backends.xnnpack.quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False

class SimplePTQuantizer:
    """
    A simple class to handle PyTorch model quantization using torch.export and torchao.
    """
    def __init__(self, model_path, log_fn=print):
        if not TORCHAO_AVAILABLE:
            raise ImportError("torchao or executorch is not installed. Please install them to use quantization.")
        
        self.log_fn = log_fn
        self.log_fn("Loading PyTorch model...")
        self.model = self.load_model(model_path)
        self.model.eval()

    def load_model(self, model_path):
        """Loads a PyTorch model, handling both state_dict and full model saves."""
        loaded = torch.load(model_path, map_location='cpu')
        if isinstance(loaded, dict):
            # This is a state dict. We need the model class definition.
            # This is a limitation for generic loaders. For now, assume a simple structure
            # that can be inferred or defined here.
            # As a placeholder, this part needs to be adapted for specific model architectures.
            # For this example, we'll raise an error.
            raise ValueError(
                "Loading from state_dict requires model class definition. "
                "Please save your model as a full object (`torch.save(model, path)`)"
            )
        else:
            return loaded

    def quantize(self, calibration_data_path=None, output_path=None, batch_size=1, input_shape=(1, 5)):
        """
        Performs post-training quantization on the loaded model.
        
        Args:
            calibration_data_path (str, optional): Path to calibration data (.pt file). 
                                                   If None, random data will be used.
            output_path (str, optional): Path to save the quantized model.
            batch_size (int): Batch size for calibration data.
            input_shape (tuple): Shape of the input tensor for random data generation.
        """
        self.log_fn("Step 1: Exporting the model graph...")
        
        # Create example inputs matching the model's expected input shape
        if calibration_data_path:
            # For now, let's assume the .pt file contains a tensor or a list of tensors
            calib_data = torch.load(calibration_data_path)
            if isinstance(calib_data, list):
                example_inputs = tuple(t.clone().detach() for t in calib_data)
            else:
                example_inputs = (calib_data.clone().detach(),)
        else:
            self.log_fn("Using random data for example inputs.")
            example_inputs = (torch.randn(input_shape),)

        try:
            # Use torch.export to capture the model graph
            exported_model_program = torch.export.export(self.model, example_inputs)
            exported_model = exported_model_program.module()
            self.log_fn("Model exported successfully.")
        except Exception as e:
            self.log_fn(f"Error during torch.export: {e}")
            self.log_fn("This model may not be compatible with torch.export.")
            return None

        self.log_fn("Step 2: Preparing model for quantization...")
        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
        prepared_model = prepare_pt2e(exported_model, quantizer)
        self.log_fn("Model prepared.")

        self.log_fn("Step 3: Calibrating the model...")
        if calibration_data_path:
            self.log_fn(f"Using calibration data from {calibration_data_path}")
            # Assuming the .pt file contains a tensor of inputs
            calib_tensor = torch.load(calibration_data_path)
            calib_dataset = TensorDataset(calib_tensor)
            data_loader = DataLoader(calib_dataset, batch_size=batch_size)
        else:
            self.log_fn("Using random data for calibration.")
            random_data = torch.randn(batch_size * 5, *input_shape[1:]) # Generate 5 batches
            dummy_labels = torch.zeros(batch_size * 5)
            dataset = TensorDataset(random_data, dummy_labels)
            data_loader = DataLoader(dataset, batch_size=batch_size)
        
        self.calibrate(prepared_model, data_loader)
        self.log_fn("Calibration complete.")

        self.log_fn("Step 4: Converting to quantized model...")
        quantized_model = convert_pt2e(prepared_model)
        self.log_fn("Model converted.")

        self.log_fn("Step 5: Saving quantized model...")
        if not output_path:
            # Create a default output path
            base, ext = os.path.splitext(example_inputs[0])
            output_path = f"{base}_quantized.pth"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(quantized_model.state_dict(), output_path)
        self.log_fn(f"Quantized model state_dict saved to: {output_path}")

        # Optional: Export to ONNX
        try:
            onnx_output_path = os.path.splitext(output_path)[0] + ".onnx"
            torch.onnx.export(quantized_model, example_inputs, onnx_output_path)
            self.log_fn(f"Quantized model also exported to ONNX: {onnx_output_path}")
        except Exception as e:
            self.log_fn(f"Could not export to ONNX: {e}")
            
        return output_path

    def calibrate(self, model, data_loader):
        """Runs calibration on the model."""
        model.eval()
        with torch.no_grad():
            for inputs, _ in data_loader:
                model(inputs)

def quantize_pytorch_model(model_path, calibration_data_path, output_path, log_fn, output_path_file=None):
    """
    High-level function to be called from Gradio UI for PyTorch quantization.
    """
    try:
        quantizer = SimplePTQuantizer(model_path=model_path, log_fn=log_fn)
        
        # We need a way to determine input shape. Let's make it a parameter for now.
        # This part might need to be more sophisticated, e.g., user input in UI.
        input_shape = (1, 3, 224, 224) # Example for an image model
        
        final_path = quantizer.quantize(
            calibration_data_path=calibration_data_path,
            output_path=output_path,
            input_shape=input_shape # Pass a default shape
        )

        if output_path_file and final_path:
            with open(output_path_file, 'w') as f:
                f.write(final_path)

        return final_path
    except Exception as e:
        log_fn(f"An error occurred: {e}")
        import traceback
        log_fn(traceback.format_exc())
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quantize a PyTorch model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the PyTorch model.")
    parser.add_argument("--calibration_data_path", type=str, help="Path to calibration data.")
    parser.add_argument("--output_path", type=str, help="Path to save the quantized model.")
    parser.add_argument("--output_path_file", type=str, help="File to write the output path to.")
    args = parser.parse_args()

    quantize_pytorch_model(args.model_path, args.calibration_data_path, args.output_path, print, args.output_path_file)
