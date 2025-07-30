#!/usr/bin/env python3
"""
🚀 NEW CONTENT GENERATOR MODEL TRAINING
Optimized training script to create a fresh, working content generation model
"""

import os
import logging
import pandas as pd
import torch
import time
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
from peft.utils import get_peft_model_state_dict
import json
import os

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from transformers.trainer_callback import TrainerCallback

class TrainingProgressCallback(TrainerCallback):
    """Custom callback to track training progress."""
    
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        logger.info(f"🚀 Starting training for {self.total_steps} steps...")
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:  # Log every 50 steps
            if self.start_time is not None:
                elapsed = time.time() - self.start_time
                progress = (state.global_step / self.total_steps) * 100
                eta = (elapsed / state.global_step) * (self.total_steps - state.global_step) if state.global_step > 0 else 0
                
                logger.info(f"📊 Step {state.global_step}/{self.total_steps} ({progress:.1f}%) - ETA: {eta/60:.1f}min")

def format_for_llama2(prompt: str, response: str) -> str:
    """Format prompt-response pairs for Llama 2 training."""
    return f"<s>[INST] {prompt} [/INST] {response} </s>"

def prepare_dataset(csv_file: str, tokenizer, max_length: int = 2048, max_samples: int = 10000):
    """Prepare dataset for training with proper formatting."""
    logger.info(f"📖 Loading dataset from {csv_file}...")
    
    # Load CSV data
    df = pd.read_csv(csv_file)
    logger.info(f"📊 Loaded {len(df)} samples")
    
    # Limit samples if needed
    if len(df) > max_samples:
        df = df.head(max_samples)
        logger.info(f"📊 Using first {max_samples} samples")
    
    # Format data for Llama 2
    formatted_data = []
    for _, row in df.iterrows():
        prompt = str(row['prompt']).strip()
        response = str(row['response']).strip()
        
        if prompt and response:  # Skip empty entries
            formatted_text = format_for_llama2(prompt, response)
            formatted_data.append({"text": formatted_text})
    
    logger.info(f"✅ Formatted {len(formatted_data)} training samples")
    
    # Create dataset
    dataset = Dataset.from_list(formatted_data)
    
    def tokenize_function(examples):
        """Tokenize the text data."""
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    logger.info(f"✅ Tokenized dataset ready for training")
    return tokenized_dataset

def create_lora_config():
    """Create ULTRA-optimized LoRA configuration for maximum speed."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Lower rank for faster inference
        lora_alpha=16,  # Lower alpha for speed
        lora_dropout=0.05,  # Lower dropout for speed
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj"
        ],  # Fewer target modules for speed
        bias="none",
        inference_mode=False
    )

def train_new_content_model(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    csv_file="advanced_prompt_to_content_dataset.csv",
    output_dir="./new_trained_content_model",
    epochs=2,
    batch_size=2,
    learning_rate=2e-4,
    max_length=2048,
    max_samples=8000
):
    """Train a new content generation model with optimized settings."""
    
    logger.info("🚀 Starting new content model training...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🖥️ Using device: {device}")
    
    if device == "cuda":
        logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load tokenizer
    logger.info("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model with 4-bit quantization
    logger.info("🤖 Loading base model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Prepare model for training
    logger.info("🔧 Preparing model for training...")
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters
    logger.info("🎯 Adding LoRA adapters...")
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Prepare dataset
    dataset = prepare_dataset(csv_file, tokenizer, max_length, max_samples)
    
    # Calculate training steps
    total_steps = (len(dataset) // batch_size) * epochs
    logger.info(f"📊 Total training steps: {total_steps}")
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # ULTRA-optimized training arguments for maximum speed
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,  # Reduced for speed
        learning_rate=learning_rate,
        fp16=True,  # Mixed precision for speed
        bf16=False,  # Disable bf16 for compatibility
        logging_steps=50,  # Less frequent logging
        save_steps=1000,  # Less frequent saving
        save_total_limit=2,  # Keep fewer checkpoints
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,
        run_name="ultra_fast_content_training",
        dataloader_pin_memory=True,  # Enable for speed
        warmup_steps=50,  # Reduced warmup
        weight_decay=0.005,  # Lower weight decay
        logging_dir=f"{output_dir}/logs",
        gradient_checkpointing=False,  # Disable for speed
        optim="adamw_torch",  # Use torch optimizer
        lr_scheduler_type="cosine",  # Better scheduler
        max_grad_norm=1.0,  # Gradient clipping
        group_by_length=True,  # Group similar lengths
        length_column_name="length",  # For grouping
        ignore_data_skip=False,
        dataloader_num_workers=0,  # Single process for stability
        torch_compile=False,  # Disable for compatibility
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[TrainingProgressCallback(total_steps)]
    )
    
    # Start training
    logger.info("🎯 Starting training...")
    trainer.train()
    
    # Save the model
    logger.info("💾 Saving trained model...")
    trainer.save_model()
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save training config
    config = {
        "model_name": model_name,
        "training_args": {k: v for k, v in training_args.to_dict().items() if not isinstance(v, set)},
        "lora_config": {k: v for k, v in lora_config.to_dict().items() if not isinstance(v, set)},
        "dataset_info": {
            "samples_used": len(dataset),
            "max_length": max_length,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        }
    }
    
    with open(f"{output_dir}/training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✅ Training complete! Model saved to {output_dir}")
    return output_dir

def main():
    """Main training function."""
    print("🚀 NEW CONTENT GENERATOR MODEL TRAINING")
    print("=" * 50)
    
    # Training configuration
    config = {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "csv_file": "advanced_prompt_to_content_dataset.csv",
        "output_dir": "./new_trained_content_model",
        "epochs": 2,
        "batch_size": 2,
        "learning_rate": 2e-4,
        "max_length": 2048,
        "max_samples": 8000
    }
    
    print(f"📊 Training Configuration:")
    print(f"   • Model: {config['model_name']}")
    print(f"   • Dataset: {config['csv_file']}")
    print(f"   • Output: {config['output_dir']}")
    print(f"   • Epochs: {config['epochs']}")
    print(f"   • Batch Size: {config['batch_size']}")
    print(f"   • Learning Rate: {config['learning_rate']}")
    print(f"   • Max Samples: {config['max_samples']}")
    
    # Estimate training time
    estimated_steps = (config['max_samples'] // config['batch_size']) * config['epochs']
    estimated_time_minutes = estimated_steps * 0.1  # Rough estimate
    
    print(f"\n⏱️ Estimated training time: {estimated_time_minutes:.1f} minutes")
    
    # Ask for confirmation
    response = input("\n🤔 Start training? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        try:
            output_dir = train_new_content_model(**config)
            print(f"\n✅ Training completed successfully!")
            print(f"📁 Model saved to: {output_dir}")
            print(f"\n🎯 To use the new model, update your content generator to use:")
            print(f"   model_path = '{output_dir}'")
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            print(f"\n❌ Training failed: {str(e)}")
    else:
        print("❌ Training cancelled.")

if __name__ == "__main__":
    main() 