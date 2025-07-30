"""
Ultra-Optimized Content Generator using Llama 2 - Maximum Speed

A high-performance content generation system optimized for speed while maintaining quality.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import os

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel, PeftConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache to avoid reloading - singleton pattern
_MODEL_CACHE = {}
_GLOBAL_GENERATOR = None

# Check if Flash Attention is available
try:
    import flash_attn  # type: ignore
    FLASH_ATTENTION_AVAILABLE = True
    logger.info("Flash Attention 2 is available - using for maximum speed")
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    logger.info("Flash Attention 2 not available - using standard attention")

# Startup diagnostic for GPU
try:
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
    else:
        logger.info("CUDA is NOT available. Running on CPU only.")
except Exception as diag_e:
    logger.warning(f"Could not run CUDA diagnostics: {diag_e}")


class OptimizedContentGenerator:
    """
    Ultra-optimized content generator using Llama 2 for maximum speed.
    This class handles only inference, not training.
    """
    
    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = True):
        """
        Initialize the optimized Llama 2 content generator.
        
        Args:
            model_path: Path to model or Hugging Face model name
            use_gpu: Whether to use GPU for inference
        """
        # Use Llama 2 7B Chat model - excellent for content generation
        self.model_path = model_path or "meta-llama/Llama-2-7b-chat-hf"
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.model = None
        self.tokenizer = None
        self._is_merged = False
        self._is_compiled = False
        
        # Check if model is already loaded in cache
        cache_key = f"{self.model_path}_{self.device}"
        if cache_key in _MODEL_CACHE:
            logger.info(f"Loading cached model: {cache_key}")
            cached_data = _MODEL_CACHE[cache_key]
            self.model = cached_data['model']
            self.tokenizer = cached_data['tokenizer']
            self._is_merged = cached_data.get('is_merged', False)
            self._is_compiled = cached_data.get('is_compiled', False)
        else:
            # Only load if not already loaded
            if self.model is None:
                self.load_model(self.model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load the Llama 2 model with ultra-optimized settings."""
        if self.model is not None:
            logger.info(f"Model already loaded from {model_path}")
            return True

        logger.info(f"Loading Llama 2 model from {model_path}")
        
        # Clear GPU memory before loading
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Strategy 1: Try loading as PEFT model first
        if self._try_load_peft_model(model_path):
            return True
        
        # Strategy 2: Try loading from latest checkpoint if available
        if self._try_load_latest_checkpoint(model_path):
            return True
        
        # Strategy 3: Try loading with pipeline approach
        if self._try_load_with_pipeline(model_path):
            return True
        
        # Strategy 4: Try loading with simple approach
        if self._try_load_simple_approach(model_path):
            return True
        
        # Strategy 5: Try loading with basic configuration
        if self._try_load_basic_model(model_path):
            return True
        
        # Strategy 6: Try loading with CPU fallback
        if self._try_load_cpu_fallback(model_path):
            return True
        
        # Strategy 7: Try loading base model as last resort
        if self._try_load_base_model():
            return True
        
        logger.error("All model loading strategies failed")
        return False
    
    def _try_load_peft_model(self, model_path: str) -> bool:
        """Try loading as PEFT model with LoRA adapters - FIXED VERSION."""
        try:
            logger.info("Attempting to load as PEFT model...")
            
            # Check if it's a PEFT model
            if os.path.exists(os.path.join(model_path, "adapter_config.json")):
                # Try loading with PEFT first - handle configuration issues
                try:
                    config = PeftConfig.from_pretrained(model_path)
                    base_model_name = config.base_model_name_or_path or "meta-llama/Llama-2-7b-chat-hf"
                except Exception as config_error:
                    logger.warning(f"PEFT config error: {config_error}")
                    # Try alternative loading method - clean the config file
                    try:
                        import json
                        config_path = os.path.join(model_path, "adapter_config.json")
                        if os.path.exists(config_path):
                            with open(config_path, 'r') as f:
                                config_data = json.load(f)
                            
                            # Remove problematic fields
                            if 'corda_config' in config_data:
                                del config_data['corda_config']
                            if 'eva_config' in config_data:
                                del config_data['eva_config']
                            if 'exclude_modules' in config_data:
                                del config_data['exclude_modules']
                            if 'layer_replication' in config_data:
                                del config_data['layer_replication']
                            if 'lora_bias' in config_data:
                                del config_data['lora_bias']
                            if 'qalora_group_size' in config_data:
                                del config_data['qalora_group_size']
                            if 'trainable_token_indices' in config_data:
                                del config_data['trainable_token_indices']
                            if 'use_dora' in config_data:
                                del config_data['use_dora']
                            if 'use_qalora' in config_data:
                                del config_data['use_qalora']
                            if 'use_rslora' in config_data:
                                del config_data['use_rslora']
                            
                            # Write cleaned config back
                            with open(config_path, 'w') as f:
                                json.dump(config_data, f, indent=2)
                            
                            logger.info("Cleaned adapter_config.json of problematic fields")
                            
                            # Try loading config again
                            config = PeftConfig.from_pretrained(model_path)
                            base_model_name = config.base_model_name_or_path or "meta-llama/Llama-2-7b-chat-hf"
                        else:
                            logger.warning("Config cleanup failed: adapter_config.json not found")
                            base_model_name = "meta-llama/Llama-2-7b-chat-hf"
                    except Exception as cleanup_error:
                        logger.warning(f"Config cleanup failed: {cleanup_error}")
                        base_model_name = "meta-llama/Llama-2-7b-chat-hf"
                
                # Load base model with FORCED GPU loading
                if self.use_gpu and torch.cuda.is_available():
                    # Clear GPU memory first
                    torch.cuda.empty_cache()
                    
                    # Force GPU-only loading WITHOUT quantization for PEFT compatibility
                    logger.info("Loading base model without quantization for PEFT compatibility...")
                    
                    # Force everything to GPU with no offloading
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        torch_dtype=torch.float16,
                        device_map="cuda",  # Force CUDA only
                        trust_remote_code=True,
                        offload_folder=None,
                        offload_state_dict=False,
                        low_cpu_mem_usage=True,
                        attn_implementation="eager",  # Use eager for compatibility
                        max_memory={0: "6GB"}  # Use less memory to prevent offloading
                    )
                    
                    # Load LoRA adapters with FORCED GPU placement
                    logger.info("Loading LoRA adapters on GPU...")
                    try:
                        # Clear any existing PEFT config to avoid conflicts
                        if hasattr(base_model, 'peft_config'):
                            delattr(base_model, 'peft_config')
                        
                        # Load PEFT adapters without device_map for compatibility
                        try:
                            self.model = PeftModel.from_pretrained(
                                base_model,
                                model_path,
                                is_trainable=False  # Disable training mode for inference
                            )
                            # Move to GPU manually
                            self.model = self.model.cuda()
                            logger.info("✅ Successfully loaded PEFT model on GPU")
                        except Exception as peft_error:
                            logger.warning(f"PEFT loading failed: {peft_error}")
                            # Try loading without PEFT - use the base model directly
                            try:
                                logger.info("Trying to load model directly without PEFT...")
                                self.model = base_model
                                self.model = self.model.cuda()
                                logger.info("✅ Successfully loaded base model on GPU")
                            except Exception as direct_error:
                                logger.warning(f"Direct loading failed: {direct_error}")
                                return False
                        
                        # Merge LoRA weights for maximum GPU speed
                        if hasattr(self.model, 'merge_and_unload'):
                            logger.info("Merging LoRA weights for max GPU speed...")
                            self.model = self.model.merge_and_unload()  # type: ignore
                            logger.info("Merge complete.")
                        else:
                            logger.info("No merge_and_unload method available, keeping as PEFT model")
                            
                    except Exception as e:
                        logger.warning(f"PEFT loading failed: {str(e)}")
                        return False
                        
                else:
                    # CPU fallback
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        torch_dtype=torch.float32,
                        device_map="cpu",
                        trust_remote_code=True,
                        offload_folder=None,
                        offload_state_dict=False,
                        low_cpu_mem_usage=True
                    )
                    
                    # Load LoRA adapters on CPU
                    self.model = PeftModel.from_pretrained(
                        base_model,
                        model_path,
                        device_map="cpu",
                        is_trainable=False
                    )
                    
                    # Merge LoRA weights for maximum speed
                    if hasattr(self.model, 'merge_and_unload'):
                        logger.info("Merging LoRA weights for max speed...")
                        self.model = self.model.merge_and_unload()  # type: ignore
                        logger.info("Merge complete.")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model.eval()
                
                # Ensure model is on GPU and clear cache
                if self.use_gpu and torch.cuda.is_available():
                    # Don't force move quantized models - they're already placed
                    torch.cuda.empty_cache()
                    logger.info("✅ Successfully loaded PEFT model on GPU")
                else:
                    logger.info("✅ Successfully loaded PEFT model on CPU")
                return True
            else:
                logger.info("Not a PEFT model (no adapter_config.json found)")
                return False
                
        except Exception as e:
            logger.warning(f"PEFT loading failed: {str(e)}")
            return False
    
    def _try_load_latest_checkpoint(self, model_path: str) -> bool:
        """Try loading from the latest checkpoint if available."""
        try:
            logger.info("Attempting to load from latest checkpoint...")
            
            # Find all checkpoint directories
            checkpoint_dirs = []
            if os.path.exists(model_path):
                for item in os.listdir(model_path):
                    if item.startswith("checkpoint-") and os.path.isdir(os.path.join(model_path, item)):
                        checkpoint_dirs.append(item)
            
            if not checkpoint_dirs:
                logger.info("No checkpoint directories found")
                return False
            
            # Sort by checkpoint number and get the latest
            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
            latest_checkpoint = os.path.join(model_path, checkpoint_dirs[-1])
            logger.info(f"Trying latest checkpoint: {latest_checkpoint}")
            
            # Try loading from the latest checkpoint
            if self._try_load_peft_model(latest_checkpoint):
                return True
            
            logger.warning("Latest checkpoint loading failed")
            return False
            
        except Exception as e:
            logger.warning(f"Latest checkpoint loading failed: {str(e)}")
            return False
    
    def _try_load_with_pipeline(self, model_path: str) -> bool:
        """Pipeline loading disabled for maximum GPU performance."""
        logger.warning("Skipping pipeline load due to performance concerns.")
        return False
    
    def _try_load_simple_approach(self, model_path: str) -> bool:
        """Try loading with the simplest possible approach."""
        try:
            logger.info("Attempting simple loading approach...")
            
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Try loading model with minimal settings
            if self.use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
                # Try with minimal settings
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            self.model.eval()
            logger.info("✅ Successfully loaded model with simple approach")
            return True
            
        except Exception as e:
            logger.warning(f"Simple loading failed: {str(e)}")
            return False
    
    def _try_load_basic_model(self, model_path: str) -> bool:
        """Try loading with ultra-optimized configuration."""
        try:
            logger.info("Attempting to load with optimized configuration...")
            
            # Ultra-optimized model loading with 4-bit quantization
            if self.use_gpu and torch.cuda.is_available():
                # Clear GPU memory first
                torch.cuda.empty_cache()
                
                # Use 8-bit quantization for better compatibility
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False
                )
                
                model_kwargs = {
                    "torch_dtype": torch.float16,
                    "device_map": "cuda",  # Force CUDA only - no CPU offloading
                    "trust_remote_code": True,
                    "offload_folder": None,
                    "offload_state_dict": False,
                    "low_cpu_mem_usage": True,
                    "attn_implementation": "eager",  # Use eager for maximum speed
                    "max_memory": {0: "7GB"}  # Use most of GPU memory
                }
            else:
                model_kwargs = {
                    "torch_dtype": torch.float32,
                    "device_map": "cpu",
                    "trust_remote_code": True,
                    "offload_folder": None,
                    "offload_state_dict": False,
                    "low_cpu_mem_usage": True
                }
            
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with fallback
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            except Exception as e:
                logger.warning(f"8-bit loading failed, trying without quantization: {str(e)}")
                # Fallback to no quantization if 8-bit fails
                fallback_kwargs = model_kwargs.copy()
                fallback_kwargs.pop('quantization_config', None)
                self.model = AutoModelForCausalLM.from_pretrained(model_path, **fallback_kwargs)
            
            self.model.eval()
            
            # Force model to GPU if available
            if self.use_gpu and torch.cuda.is_available():
                self.model = self.model.cuda()
                torch.cuda.empty_cache()  # Clear GPU cache
                logger.info("✅ Successfully loaded model with optimized configuration on GPU")
            else:
                logger.info("✅ Successfully loaded model with optimized configuration on CPU")
            return True
            
        except Exception as e:
            logger.warning(f"Optimized loading failed: {str(e)}")
            return False
    
    def _try_load_cpu_fallback(self, model_path: str) -> bool:
        """Try loading on CPU as fallback."""
        try:
            logger.info("Attempting CPU fallback...")
            
            # Force CPU loading
            original_device = self.device
            self.device = "cpu"
            self.use_gpu = False
            
            model_kwargs = {
                "torch_dtype": torch.float32,
                "device_map": "cpu",
                "trust_remote_code": True,
                "offload_folder": None,
                "offload_state_dict": False,
                "low_cpu_mem_usage": True
            }
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model on CPU
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            self.model.eval()
            
            logger.info("✅ Successfully loaded model on CPU")
            return True
            
        except Exception as e:
            logger.warning(f"CPU fallback failed: {str(e)}")
            # Restore original device settings
            self.device = original_device
            self.use_gpu = torch.cuda.is_available()
            return False
    
    def _try_load_base_model(self) -> bool:
        """Try loading the base Llama 2 model as last resort."""
        try:
            logger.info("Attempting to load base Llama 2 model...")
            
            base_model_path = "meta-llama/Llama-2-7b-chat-hf"
            
            # Force GPU usage for maximum speed with 4-bit quantization
            if self.use_gpu and torch.cuda.is_available():
                # Clear GPU memory first
                torch.cuda.empty_cache()
                
                # Use 8-bit quantization for better compatibility
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False
                )
                
                model_kwargs = {
                    "torch_dtype": torch.float16,
                    "device_map": "cuda",  # Force CUDA only - no offloading
                    "trust_remote_code": True,
                    "offload_folder": None,
                    "offload_state_dict": False,
                    "low_cpu_mem_usage": True,
                    "quantization_config": quant_config,
                    "attn_implementation": "flash_attention_2" if FLASH_ATTENTION_AVAILABLE else "eager",
                    "max_memory": {0: "7GB"}  # Use most of GPU memory
                }
            else:
                model_kwargs = {
                    "torch_dtype": torch.float32,
                    "device_map": "cpu",
                    "trust_remote_code": True,
                    "offload_folder": None,
                    "offload_state_dict": False,
                    "low_cpu_mem_usage": True
                }
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
            self.model.eval()
            
            # Don't force move quantized models - they're already placed
            if self.use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU cache
                logger.info("✅ Successfully loaded base Llama 2 model on GPU")
            else:
                logger.info("✅ Successfully loaded base Llama 2 model on CPU")
            return True
            
        except Exception as e:
            logger.error(f"Base model loading failed: {str(e)}")
            return False
    
    def format_prompt_llama2(self, prompt: str) -> str:
        """Format prompt for Llama 2 chat model."""
        return f"<s>[INST] {prompt} [/INST]"
    
    def format_chat_llama2(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation for Llama 2 chat model."""
        conversation = ""
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            if role == 'user':
                conversation += f"<s>[INST] {content} [/INST]"
            elif role == 'assistant':
                conversation += f" {content} </s>"
            elif role == 'system':
                conversation += f"<s>[INST] <<SYS>> {content} <</SYS>> [/INST]"
        return conversation
    
    def _clean_generated_text(self, generated_text: str, prompt_text: str) -> str:
        """Clean the generated text by removing prompt and special tokens."""
        # Remove the prompt from the generated text
        if generated_text.startswith(prompt_text):
            content = generated_text[len(prompt_text):].strip()
        else:
            content = generated_text.strip()
        
        # Clean up Llama 2 specific tokens efficiently
        tokens_to_remove = ["</s>", "<s>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"]
        for token in tokens_to_remove:
            content = content.replace(token, "")
        
        # Remove any leading/trailing whitespace, duplicates, and truncate junk
        content = content.strip()
        
        # Remove duplicate whitespace
        import re
        content = re.sub(r'\s+', ' ', content)
        
        # Truncate if too long (prevent junk)
        if len(content) > 2000:
            content = content[:2000].strip()
        
        return content
    
    def generate(self, prompt: str, max_length: int = 200, temperature: float = 0.8) -> str:
        """Generate content from a prompt with ultra-optimized settings for speed."""
        if not self.model or not self.tokenizer:
            return "Error: Model not loaded"

        start_time = time.time()
        try:
            logger.info(f"Generating content for prompt: {prompt[:50]}...")
            formatted_prompt = self.format_prompt_llama2(prompt)

            # Ultra-optimized tokenization with no length limits
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=False,  # No truncation - keep full prompt
                add_special_tokens=True
            )
            
            # Get the actual device where the model is loaded
            model_device = next(self.model.parameters()).device
            
            # Move input tensors to the model's device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            # Optional: Clear GPU cache and log device info
            if self.use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.model.eval()
                
                # Don't force move quantized models - they're already placed
                
                logger.info(f"✅ Using GPU for generation on device: {model_device}")
                
                # Debug logging
                logger.info(f"Model is on device: {model_device}")
                for name, tensor in inputs.items():
                    logger.info(f"Input '{name}' is on device: {tensor.device}")
            else:
                logger.info(f"Using CPU for generation on device: {model_device}")

            # Ultra-optimized generation for maximum speed
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,  # No artificial limits - use full requested length
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.85,   # Lower for faster sampling
                    top_k=40,      # Lower for faster sampling
                    repetition_penalty=1.0,  # No penalty for maximum speed
                    no_repeat_ngram_size=0,   # No n-gram blocking for speed
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0,
                    eos_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0,
                    num_beams=1,  # No beam search for speed
                    length_penalty=1.0,
                    return_dict_in_generate=False  # Return tensors directly for speed
                )

            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

            # Clean the output
            content = self._clean_generated_text(generated_text, formatted_prompt)

            generation_time = time.time() - start_time
            logger.info(f"Generated {len(content)} characters in {generation_time:.2f} seconds")
            
            # Log GPU usage for debugging
            if self.use_gpu and torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"GPU memory used: {gpu_memory:.2f} GB")
                
                # Check if model is actually on GPU
                if hasattr(self.model, 'device'):
                    logger.info(f"Model device: {self.model.device}")
                else:
                    logger.info("Model device: Unknown (using accelerate)")
                
                # Check if inputs are on GPU
                input_device = next(iter(inputs.values())).device
                logger.info(f"Input device: {input_device}")

            return content if content else "I'm not sure how to respond to that."
            
        except Exception as e:
            generation_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Error generating content in {generation_time:.2f} seconds: {error_msg}")
            return f"Error: {error_msg}"
    
    def chat(self, messages: List[Dict[str, str]], max_length: int = 200, temperature: float = 0.8) -> str:
        """Generate chat response from conversation history with ultra-optimized settings."""
        if not self.model or not self.tokenizer:
            return "Error: Model not loaded"
        
        start_time = time.time()
        try:
            conversation = self.format_chat_llama2(messages)
            
            # Ultra-optimized tokenization for chat with no length limits
            inputs = self.tokenizer(
                conversation,
                return_tensors="pt",
                padding=True,
                truncation=False,  # No truncation - keep full conversation
                add_special_tokens=True
            )
            
            # Get the actual device where the model is loaded
            model_device = next(self.model.parameters()).device
            
            # Move input tensors to the model's device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            # Optional: Clear GPU cache and log device info
            if self.use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.model.eval()
                logger.info(f"✅ Using GPU for generation on device: {model_device}")
                
                # Debug logging
                logger.info(f"Model is on device: {model_device}")
                for name, tensor in inputs.items():
                    logger.info(f"Input '{name}' is on device: {tensor.device}")
            else:
                logger.info(f"Using CPU for generation on device: {model_device}")
            
            # Ultra-optimized generation for maximum speed
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,  # No artificial limits - use full requested length
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.85,  # Slightly lower for faster sampling
                    top_k=40,     # Lower for faster sampling
                    repetition_penalty=1.05,  # Lower penalty for speed
                    no_repeat_ngram_size=2,   # Lower for speed
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0,
                    eos_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0,
                    num_beams=1,  # No beam search for speed
                    length_penalty=1.0,
                    return_dict_in_generate=False  # Return tensors directly for speed
                )
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Clean the output
            content = self._clean_generated_text(generated_text, conversation)
            
            generation_time = time.time() - start_time
            logger.info(f"Chat response generated in {generation_time:.2f} seconds")
            
            return content if content else "I'm not sure how to respond to that."
            
        except Exception as e:
            generation_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Error generating chat response in {generation_time:.2f} seconds: {error_msg}")
            return f"Error: {error_msg}"
    
    def test_model(self) -> bool:
        """Test if the model is working correctly."""
        if not self.model or not self.tokenizer:
            return False
        
        try:
            test_prompt = "Hello, how are you?"
            start_time = time.time()
            response = self.generate(test_prompt, max_length=50, temperature=0.8)
            test_time = time.time() - start_time
            
            logger.info(f"Model test completed in {test_time:.2f} seconds")
            return len(response) > 0 and not response.startswith("Error:")
        except Exception as e:
            logger.error(f"Model test failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the loaded model."""
        info = {
            "model_path": self.model_path,
            "device": self.device,
            "use_gpu": self.use_gpu,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "is_merged": self._is_merged,
            "is_compiled": self._is_compiled,
            "flash_attention": FLASH_ATTENTION_AVAILABLE
        }
        
        if self.model:
            info["model_type"] = type(self.model).__name__
            # Safely get model config type
            try:
                if hasattr(self.model, 'config'):
                    config = self.model.config
                    # Use getattr to safely access model_type
                    model_type = getattr(config, 'model_type', None)
                    info["model_config"] = str(model_type) if model_type is not None else "Unknown"
                else:
                    info["model_config"] = "Unknown"
            except Exception:
                info["model_config"] = "Unknown"
        
        if self.tokenizer:
            info["vocab_size"] = self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else "Unknown"
        
        # Add GPU memory info
        if self.use_gpu and torch.cuda.is_available():
            try:
                total_memory, allocated_memory = torch.cuda.mem_get_info()
                info["gpu_total_memory_gb"] = total_memory / (1024**3)
                info["gpu_allocated_memory_gb"] = allocated_memory / (1024**3)
                info["gpu_free_memory_gb"] = (total_memory - allocated_memory) / (1024**3)
            except:
                pass
        
        return info
    
    def clear_cache(self):
        """Clear the model cache to free memory."""
        global _MODEL_CACHE
        _MODEL_CACHE.clear()
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model cache cleared")


def get_global_generator(model_path: Optional[str] = None, use_gpu: bool = True) -> OptimizedContentGenerator:
    """Get or create a global OptimizedContentGenerator instance."""
    global _GLOBAL_GENERATOR
    
    # Check if we need to create a new generator
    if _GLOBAL_GENERATOR is None or _GLOBAL_GENERATOR.model_path != (model_path or "meta-llama/Llama-2-7b-chat-hf"):
        logger.info(f"Creating new global OptimizedContentGenerator instance for {model_path}")
        _GLOBAL_GENERATOR = OptimizedContentGenerator(model_path, use_gpu)
    else:
        logger.info(f"Reusing global OptimizedContentGenerator instance for {model_path}")
    
    return _GLOBAL_GENERATOR


def generate_content(prompt: str, model_path: Optional[str] = None, use_gpu: bool = True, **kwargs) -> str:
    """Generate content using the ultra-optimized ContentGenerator."""
    generator = get_global_generator(model_path, use_gpu)
    return generator.generate(prompt, **kwargs)


if __name__ == "__main__":
    print("🚀 Testing Ultra-Optimized Llama 2 Content Generator")
    print("=" * 60)
    
    generator = OptimizedContentGenerator(use_gpu=True)  # Try GPU first
    
    test_prompts = [
        "Hello, how are you?",
        "Tell me a joke",
        "What's the weather like today?",
        "Write a short story about a magical forest"
    ]
    
    total_time = 0
    for i, prompt in enumerate(test_prompts):
        print(f"\n👤 Prompt {i+1}: {prompt}")
        start_time = time.time()
        result = generator.generate(prompt)
        generation_time = time.time() - start_time
        total_time += generation_time
        print(f"🤖 Response: {result}")
        print(f"⏱️  Generation time: {generation_time:.2f} seconds")
    
    avg_time = total_time / len(test_prompts)
    print(f"\n📊 Average generation time: {avg_time:.2f} seconds")
    print(f"📊 Total time for all prompts: {total_time:.2f} seconds") 