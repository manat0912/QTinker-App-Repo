
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Assuming distillation strategies from distillation.py are available
# You might need to adjust imports based on your project structure
from distillation import (
    LogitKD, PatientKD, FeatureMatchingKD,
    AttentionDistillationKD, MultiTeacherKD
)

STRATEGY_MAP = {
    "LogitKD": LogitKD,
    "PatientKD": PatientKD,
    "FeatureMatchingKD": FeatureMatchingKD,
    "AttentionDistillationKD": AttentionDistillationKD,
    "MultiTeacherKD": MultiTeacherKD
}

class Distiller:
    """
    A class to handle the distillation process for PyTorch models,
    specifically tailored for models from the transformers library.
    """
    def __init__(self, teacher_model_path, student_model_path, strategy_name, log_fn=print, custom_model_path=None):
        self.log_fn = log_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_fn(f"Using device: {self.device}")

        self.log_fn("Loading models...")
        self.teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path).to(self.device)
        self.student_model = AutoModelForCausalLM.from_pretrained(student_model_path, trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)

        self.custom_model = None
        if custom_model_path and strategy_name == "MultiTeacherKD":
            self.log_fn("Loading custom model for Multi-Teacher Distillation...")
            self.custom_model = AutoModelForCausalLM.from_pretrained(custom_model_path).to(self.device)
        
        self.teacher_model.eval() # Teacher model is always in eval mode

        self.strategy = self.get_strategy(strategy_name)
        self.log_fn(f"Using distillation strategy: {strategy_name}")

    def get_strategy(self, strategy_name):
        """Initializes and returns the selected distillation strategy."""
        if strategy_name not in STRATEGY_MAP:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        strategy_class = STRATEGY_MAP[strategy_name]
        
        # These are example initializations. You may need to pass more specific params
        # from the UI, like which layers to match for PatientKD.
        if strategy_name == "PatientKD":
            # Example: matching the middle and last layers. This should be configurable.
            teacher_num_layers = self.teacher_model.config.num_hidden_layers
            student_num_layers = self.student_model.config.num_hidden_layers
            teacher_layers = [teacher_num_layers // 2, teacher_num_layers]
            student_layers = [student_num_layers // 2, student_num_layers]
            return strategy_class(self.teacher_model, self.student_model, student_layers=student_layers, teacher_layers=teacher_layers)
        elif strategy_name == "MultiTeacherKD":
            if not self.custom_model:
                raise ValueError("MultiTeacherKD requires a custom model.")
            return strategy_class(self.teacher_model, self.student_model, self.custom_model)
        
        return strategy_class(self.teacher_model, self.student_model)

    def distill(self, data_path, output_dir, epochs=3, learning_rate=5e-5, batch_size=4):
        """
        Runs the distillation process.
        """
        self.log_fn("Preparing data...")
        # For demonstration, we'll use a simple text file as data.
        # In a real scenario, you would use a proper dataset (e.g., from Hugging Face datasets)
        with open(data_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
        data_loader = DataLoader(dataset, batch_size=batch_size)

        optimizer = optim.Adam(self.student_model.parameters(), lr=learning_rate)
        
        self.log_fn("Starting distillation...")
        self.student_model.train()

        for epoch in range(epochs):
            total_loss = 0
            for batch in data_loader:
                optimizer.zero_grad()
                
                input_ids, attention_mask = [b.to(self.device) for b in batch]
                
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(input_ids, attention_mask=attention_mask, output_hidden_states=True, output_attentions=True)
                    custom_outputs = None
                    if self.custom_model:
                         custom_outputs = self.custom_model(input_ids, attention_mask=attention_mask, output_hidden_states=True, output_attentions=True)

                student_outputs = self.student_model(input_ids, attention_mask=attention_mask, output_hidden_states=True, output_attentions=True)
                
                loss = self.strategy.compute_loss(student_outputs, teacher_outputs, custom_outputs)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            self.log_fn(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader)}")

        self.log_fn("Distillation complete. Saving model...")
        
        os.makedirs(output_dir, exist_ok=True)
        self.student_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        self.log_fn(f"Distilled student model saved to: {output_dir}")
        return output_dir

def run_pytorch_distillation(teacher_path, student_path, data_path, strategy, output_dir, custom_path=None, log_fn=print):
    """
    High-level function to be called from the Gradio UI for PyTorch distillation.
    """
    try:
        distiller = Distiller(
            teacher_model_path=teacher_path,
            student_model_path=student_path,
            custom_model_path=custom_path,
            strategy_name=strategy,
            log_fn=log_fn
        )
        distilled_path = distiller.distill(data_path, output_dir)
        return distilled_path
    except Exception as e:
        log_fn(f"An error occurred during distillation: {e}")
        import traceback
        log_fn(traceback.format_exc())
        return None
