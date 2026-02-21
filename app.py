"""
Main entry point for the application (Pinokio compatible).
"""
import os
import glob
import sys
import logging

# This ensures the script can find packages installed in the virtual environment.
venv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'env', 'Lib', 'site-packages'))
if venv_path not in sys.path:
    sys.path.insert(0, venv_path)

# Suppress the specific warning from torch_tensorrt
# This is a cleaner way to handle noisy library warnings without disabling all logging
logging.getLogger("torch_tensorrt").setLevel(logging.ERROR)

from gradio_ui import create_ui, custom_theme, css

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=custom_theme, css=css)
