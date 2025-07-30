# Llama 2 Content Generator with Training Support

A high-quality content generation system using Llama 2 with the ability to fine-tune on custom datasets for specific requirements.

## Features

- **Llama 2 Integration**: Uses the powerful Llama 2 7B Chat model
- **Custom Training**: Fine-tune the model on your own dataset
- **Fast Generation**: Optimized for speed with model caching
- **Interactive Chat**: Real-time conversation mode
- **GPU/CPU Support**: Works on both GPU and CPU
- **Memory Efficient**: Includes optimizations for limited memory

## Installation

1. **Install Python dependencies:**
```bash
pip install -r ../requirements.txt
```

2. **For GPU support (recommended):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **Optional: Install flash-attention for better performance (Linux/Mac only):**
```bash
pip install flash-attn
```

## Quick Start

### 1. Basic Content Generation
```bash
# Generate content from a prompt
python main.py generate --prompt "Hello, how are you?"

# Generate with custom settings
python main.py generate --prompt "Tell me a joke" --max-length 300 --temperature 0.8
```

### 2. Interactive Chat Mode
```bash
# Start interactive chat
python main.py chat

# Use CPU mode if you have memory issues
python main.py chat --cpu-only
```

### 3. Test the Model
```bash
# Test if everything is working
python main.py test
```

## Training on Custom Data

### 1. Create Training Dataset

First, create a sample dataset to understand the format:
```bash
python main.py create-dataset --output-file training_data.json
```

This creates a JSON file with the required format:
```json
[
  {
    "prompt": "What is machine learning?",
    "response": "Machine learning is a subset of artificial intelligence..."
  },
  {
    "prompt": "Tell me a joke about programming",
    "response": "Why do programmers prefer dark mode? Because light attracts bugs!"
  }
]
```

### 2. Prepare Your Custom Dataset

Create your own JSON file with the same format. Include:
- **prompt**: The input text/prompt
- **response**: The expected output/response

Example for food-related content:
```json
[
  {
    "prompt": "How do I make pasta carbonara?",
    "response": "To make pasta carbonara, you'll need spaghetti, eggs, pancetta, pecorino cheese, and black pepper. Cook the pasta, crisp the pancetta, mix eggs and cheese, then combine everything while the pasta is hot."
  },
  {
    "prompt": "What are the health benefits of broccoli?",
    "response": "Broccoli is rich in vitamins C and K, fiber, and antioxidants. It supports immune health, bone health, and may help reduce inflammation and cancer risk."
  }
]
```

### 3. Train Your Model

```bash
# Basic training
python main.py train --data-file your_data.json --output-dir ./my_trained_model

# Advanced training with custom parameters
python main.py train \
  --data-file your_data.json \
  --output-dir ./my_trained_model \
  --epochs 5 \
  --batch-size 2 \
  --learning-rate 1e-5
```

### 4. Use Your Trained Model

```bash
# Generate content with your trained model
python main.py generate --prompt "Your prompt" --model-path ./my_trained_model

# Interactive chat with your trained model
python main.py chat --model-path ./my_trained_model
```

## Command Line Options

### Generate Command
```bash
python main.py generate [OPTIONS]

Options:
  --prompt TEXT           Your prompt (required)
  --model-path TEXT       Path to model or Hugging Face model name
  --max-length INTEGER    Maximum response length (default: 200)
  --temperature FLOAT     Creativity level 0.1-1.0 (default: 0.7)
  --cpu-only              Use CPU only (for memory issues)
```

### Chat Command
```bash
python main.py chat [OPTIONS]

Options:
  --model-path TEXT       Path to model or Hugging Face model name
  --max-length INTEGER    Maximum response length (default: 200)
  --temperature FLOAT     Creativity level 0.1-1.0 (default: 0.7)
  --cpu-only              Use CPU only (for memory issues)
```

### Train Command
```bash
python main.py train [OPTIONS]

Options:
  --data-file TEXT        Path to JSON training data file (required)
  --output-dir TEXT       Output directory for trained model (default: ./trained_model)
  --model-path TEXT       Base model to fine-tune (default: meta-llama/Llama-2-7b-chat-hf)
  --epochs INTEGER        Number of training epochs (default: 3)
  --batch-size INTEGER    Training batch size (default: 4)
  --learning-rate FLOAT   Learning rate (default: 2e-5)
```

### Create Dataset Command
```bash
python main.py create-dataset --output-file training_data.json
```

### Test Command
```bash
python main.py test [OPTIONS]

Options:
  --model-path TEXT       Path to model or Hugging Face model name
  --cpu-only              Use CPU only (for memory issues)
```

## Interactive Chat Commands

When in chat mode, you can use these commands:
- `quit` or `exit`: Leave the chat
- `help`: Show available commands
- `clear`: Clear conversation history
- `cache`: Clear model cache (frees memory)
- `info`: Show model information
- `stats`: Show session statistics

## System Requirements

### Minimum Requirements
- **RAM**: 8GB (16GB recommended)
- **Storage**: 20GB free space
- **Python**: 3.8+

### Recommended Requirements
- **RAM**: 16GB+ for training, 8GB+ for inference
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for faster training)
- **Storage**: 50GB+ free space (for model downloads and training)

### Memory Optimization
If you encounter memory issues:
1. Use `--cpu-only` flag
2. Reduce batch size during training
3. Use smaller max_length values
4. Clear model cache with `cache` command in chat mode

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   ```bash
   # Use CPU mode
   python main.py generate --prompt "Hello" --cpu-only
   ```

2. **Model Download Issues**
   - Check internet connection
   - Ensure you have enough disk space
   - Try downloading during off-peak hours

3. **Training Errors**
   - Reduce batch size: `--batch-size 2`
   - Use CPU mode: Add `--cpu-only` to training
   - Check data format in your JSON file

4. **Slow Generation**
   - Use GPU if available
   - Reduce max_length
   - Model is cached after first use for faster subsequent runs

### Performance Tips

1. **For Faster Training:**
   - Use GPU with sufficient VRAM
   - Increase batch size if memory allows
   - Use gradient accumulation for larger effective batch sizes

2. **For Better Quality:**
   - Use more training data (1000+ examples)
   - Train for more epochs (5-10)
   - Experiment with learning rates

3. **For Memory Efficiency:**
   - Use CPU mode if GPU memory is limited
   - Reduce batch size during training
   - Clear cache when switching between models

## Examples

### Food Content Generation
```bash
# Train on food-related data
python main.py train --data-file food_data.json --output-dir ./food_model

# Generate food content
python main.py generate --prompt "How do I make chocolate cake?" --model-path ./food_model
```

### Technical Content
```bash
# Train on technical documentation
python main.py train --data-file tech_data.json --output-dir ./tech_model

# Generate technical explanations
python main.py generate --prompt "Explain machine learning" --model-path ./tech_model
```

### Creative Writing
```bash
# Train on creative writing samples
python main.py train --data-file creative_data.json --output-dir ./creative_model

# Generate creative content
python main.py generate --prompt "Write a story about a magical forest" --model-path ./creative_model
```

## Model Information

- **Base Model**: Llama 2 7B Chat (meta-llama/Llama-2-7b-chat-hf)
- **Model Size**: 7B parameters
- **Training**: Supports fine-tuning on custom datasets
- **Optimization**: 8-bit quantization, model caching
- **Format**: Chat format with [INST] and [/INST] tags

## License

This project uses Llama 2, which requires accepting the Meta license. Please ensure you comply with the model's license terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the command examples
3. Ensure your data format is correct
4. Check system requirements

The model is designed to be user-friendly and provides helpful error messages to guide you through any issues. 