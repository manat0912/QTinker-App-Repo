"""
Knowledge Distillation implementation with teacher-student training.
"""
import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple

from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer, AutoModel
from core.device_manager import get_device_manager


class PromptDataset(Dataset):
    """Dataset for loading training prompts from a text file."""
    
    def __init__(self, tokenizer, file_path: str, max_length: int = 256):
        """
        Initialize dataset.
        
        Args:
            tokenizer: Tokenizer to use
            file_path: Path to text file with one prompt per line
            max_length: Maximum sequence length
        """
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        # Load prompts from file
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                self.lines = [l.strip() for l in f.readlines() if l.strip()]
        else:
            # Default prompts if file doesn't exist
            self.lines = [
                "Explain quantum computing in simple terms.",
                "Write a short story about a dragon.",
                "Summarize the causes of World War I.",
            ]
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        text = self.lines[idx]
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        # Flatten batch dimension
        return {k: v.squeeze(0) for k, v in enc.items()}


def load_config() -> dict:
    """Load settings from YAML config."""
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def load_paths_config() -> dict:
    """Load paths from YAML config."""
    config_path = Path(__file__).parent.parent / "config" / "paths.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {"train_prompts": "data/train_prompts.txt"}


def distill_teacher_student(
    student: torch.nn.Module,
    student_tokenizer,
    teacher_model_path: str,
    teacher_type: str,
    device_manager,
    log_fn=None,
    epochs: int = 50,
    batch_size: int = 4,
    max_steps: int = 100,
    learning_rate: float = 5e-5,
    temperature: float = 1.0,
    alpha: float = 0.5,
    teacher_arch: str = "Causal LM"
) -> torch.nn.Module:
    
    learning_rate = float(learning_rate)
    temperature = float(temperature)
    alpha = float(alpha)
    epochs = int(epochs)
    batch_size = int(batch_size)
    max_steps = int(max_steps)

    if log_fn:
        log_fn("=" * 50)
        log_fn("Starting Teacher-Student Knowledge Distillation")
        log_fn("=" * 50)
    
    device = device_manager.get_device()
    
    if log_fn:
        log_fn(f"Loading teacher model from: {teacher_model_path}")
    
    try:
        is_hf_folder = os.path.isdir(teacher_model_path)
        
        if is_hf_folder or teacher_type == "HuggingFace folder":
            if log_fn:
                log_fn("Detected HuggingFace folder structure...")
                
            if teacher_arch == "Masked LM (BERT, RoBERTa)":
                teacher = AutoModelForMaskedLM.from_pretrained(
                    teacher_model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="cpu"
                )
            elif teacher_arch == "Generic AutoModel":
                teacher = AutoModel.from_pretrained(
                    teacher_model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="cpu"
                )
            else:
                teacher = AutoModelForCausalLM.from_pretrained(
                    teacher_model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="cpu"
                )
        else:
            if log_fn:
                log_fn("Loading as single torch file...")
            teacher = torch.load(teacher_model_path, map_location="cpu", weights_only=False)
            
    except Exception as e:
        if log_fn:
            log_fn(f"ERROR loading teacher model: {e}")
        raise e
    
    teacher = device_manager.move_model_to_device(teacher)
    student = device_manager.move_model_to_device(student)
    
    teacher.eval()
    student.train()
    
    paths_config = load_paths_config()
    train_prompts_path = paths_config.get("train_prompts", "data/train_prompts.txt")
    train_prompts_path = Path(__file__).parent.parent / train_prompts_path
    
    dataset = PromptDataset(student_tokenizer, str(train_prompts_path))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate)
    
    step = 0
    
    for epoch in range(epochs):
        if step >= max_steps:
            break
        if log_fn:
            log_fn(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        for batch_idx, batch in enumerate(loader):
            if step >= max_steps:
                break
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            with torch.no_grad():
                teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = getattr(teacher_outputs, "logits", None)
                if teacher_logits is None:
                    teacher_logits = teacher_outputs.last_hidden_state

            student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = getattr(student_outputs, "logits", None)
            if student_logits is None:
                student_logits = student_outputs.last_hidden_state

            seq_len = min(teacher_logits.size(1), student_logits.size(1))
            teacher_logits = teacher_logits[:, :seq_len, :].to(torch.float32)
            student_logits = student_logits[:, :seq_len, :].to(torch.float32)
            
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            
            kl_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="batchmean"
            ) * (temperature ** 2)
            
            logits_flat = student_logits.view(-1, student_logits.size(-1))
            labels_flat = input_ids[:, :seq_len].reshape(-1)
            mask_flat = attention_mask[:, :seq_len].reshape(-1).to(torch.float32)
            
            ce_loss_raw = F.cross_entropy(logits_flat, labels_flat, reduction="none")
            ce_loss = (ce_loss_raw * mask_flat).sum() / (mask_flat.sum() + 1e-9)
            
            loss = alpha * kl_loss + (1 - alpha) * ce_loss

            if torch.isnan(loss) or loss > 10000:
                optimizer.zero_grad()
                continue
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            
            step += 1
            
            if log_fn:
                log_fn(f"Step {step}/{max_steps} - Loss: {loss.item():.4f} (KL: {kl_loss.item():.4f}, CE: {ce_loss.item():.4f})")

    student.eval()
    student.to("cpu")
    teacher.to("cpu")
    device_manager.clear_cache()
    
    if log_fn:
        log_fn("=" * 50)
        log_fn("Knowledge Distillation Complete!")
        log_fn("=" * 50)
    
    return student

def distill_model(
    model: torch.nn.Module,
    tokenizer,
    device_manager=None,
    log_fn=None,
    distillation_config: dict = None
) -> torch.nn.Module:
    """
    Distill a model using either placeholder or teacher-student method.
    
    Args:
        model: The student model to distill
        tokenizer: Model tokenizer
        device_manager: Device manager instance
        log_fn: Optional logging function
        distillation_config: Distillation configuration dict
        
    Returns:
        Distilled model
    """
    if device_manager is None:
        device_manager = get_device_manager(log_fn)
    
    # Load config if not provided
    if distillation_config is None:
        config = load_config()
        distillation_config = config.get("distillation", {})
    
    if not distillation_config.get("enabled", True):
        if log_fn:
            log_fn("Distillation disabled, skipping...")
        return model
    
    mode = distillation_config.get("mode", "placeholder")
    
    if mode == "placeholder":
        if log_fn:
            log_fn("Running placeholder distillation (no training)...")
            log_fn(f"Distillation running on: {device_manager.get_device_name()}")
        
        # Ensure model is on the right device
        try:
            current_device = next(model.parameters()).device
            if current_device != device_manager.get_device():
                if log_fn:
                    log_fn(f"Moving model to {device_manager.get_device_name()} for distillation...")
                try:
                    model = device_manager.move_model_to_device(model)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        if log_fn:
                            log_fn("⚠️  GPU OOM during distillation. Using CPU...")
                        device_manager.switch_to_cpu()
                        model = model.to(torch.device("cpu"))
        except StopIteration:
            if log_fn:
                log_fn("Model has no parameters, skipping device placement for distillation.")
        
        device_manager.clear_cache()
        return model
    
    elif mode == "teacher_student":
        teacher_path = distillation_config.get("teacher_model_path", "")
        if not teacher_path:
            if log_fn:
                log_fn("⚠️  Teacher model path not specified, using placeholder mode")
            return distill_model(model, tokenizer, device_manager, log_fn, {"mode": "placeholder", "enabled": True})
        
        return distill_teacher_student(
            student=model,
            student_tokenizer=tokenizer,
            teacher_model_path=teacher_path,
            teacher_type=distillation_config.get("teacher_type", "HuggingFace folder"),
            device_manager=device_manager,
            log_fn=log_fn,
            epochs=distillation_config.get("epochs", 1),
            batch_size=distillation_config.get("batch_size", 4),
            max_steps=distillation_config.get("max_steps", 100),
            learning_rate=distillation_config.get("learning_rate", 5e-5),
            temperature=distillation_config.get("temperature", 1.0),
            alpha=distillation_config.get("alpha", 0.5),
            teacher_arch=distillation_config.get("teacher_arch", "Causal LM")
        )
    
    else:
        raise ValueError(f"Unknown distillation mode: {mode}")
