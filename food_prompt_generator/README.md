# 🚀 Intelligent Food Content Generation System

A sophisticated two-stage content generation system that combines fine-tuned LLaMA 2 models with semantic embeddings and intelligent template fallbacks.

**Stage 1**: Prompt Generator → Creates rich, detailed prompts  
**Stage 2**: Content Generator → Generates structured content from prompts

## 📦 Project Structure

```
food_prompt_generator/
├── 📁 portable_prompt_generator/     # 🎯 PORTABLE PROMPT MODEL (Use this!)
│   ├── portable_model.py            # Main portable model
│   ├── config.json                  # Configuration
│   ├── requirements.txt             # Dependencies
│   ├── README.md                    # Documentation
│   ├── example.py                   # Usage examples
│   ├── test.py                      # Test script
│   ├── api_server.py                # API server
│   └── model/                       # Trained model files
├── 📁 content_generator/            # 🆕 CONTENT GENERATION MODEL
│   ├── content_generator.py         # Main content generator
│   ├── data_preparation.py          # Dataset creation
│   ├── training_script.py           # Model training
│   ├── pipeline.py                  # Complete pipeline
│   ├── main.py                      # CLI interface
│   ├── requirements.txt             # Dependencies
│   ├── README.md                    # Documentation
│   └── test_setup.py                # Setup verification
├── prompt_generator.py              # Main development model
├── main.py                          # CLI interface
├── requirements.txt                 # Development dependencies
├── setup_portable.py                # Setup script
├── create_portable_package.py       # Package creation script
├── PORTABLE_MODEL_GUIDE.md          # Complete deployment guide
├── DEPLOYMENT_GUIDE.md              # Comprehensive deployment guide
└── INTEGRATION_SUMMARY.md           # Integration options
```

## 🎯 Quick Start

### **Stage 1: Prompt Generator (Ready to Use)**
```bash
# Use the portable prompt model
cd portable_prompt_generator
pip install -r requirements.txt
python test.py
```

### **Stage 2: Content Generator (Trainable)**
```bash
# Set up content generation system
cd content_generator
pip install -r requirements.txt
python test_setup.py

# Create training dataset
python main.py create-dataset --num-samples 200

# Train the model
python main.py train --num-samples 500 --output-dir ./content_generator_model

# Test the model
python main.py test --model-path ./content_generator_model
```

### **Complete Pipeline (Both Stages)**
```bash
# Run complete content generation from food name
python content_generator/main.py pipeline --food-name "Chocolate Cake" --content-type recipe --style romantic
```

## 🚀 Two-Stage System

### **Stage 1: Portable Prompt Model (Ready to Use)**

The `portable_prompt_generator/` folder contains a complete, standalone version of your prompt generator:

#### **Features:**
- ✅ **Portable** - Works on any system
- ✅ **Reliable** - Always has fallback mode
- ✅ **Complete** - Includes trained model files
- ✅ **Ready-to-use** - Test script and API server included

#### **Usage:**
```python
from portable_model import generate_prompt

keywords = ["coffee", "morning", "peace"]
prompt, theme, tone = generate_prompt(keywords, "lifestyle", "post")
print(f"Prompt: {prompt}")
```

#### **API Server:**
```bash
cd portable_prompt_generator
python api_server.py
# Server runs on http://localhost:5000
```

### **Stage 2: Content Generator (Trainable)**

The `content_generator/` folder contains a complete content generation system:

#### **Features:**
- ✅ **Trainable** - Can be fine-tuned on your data
- ✅ **Multiple Content Types** - Recipes, descriptions, blog posts, etc.
- ✅ **Multiple Styles** - Professional, casual, romantic, elegant, fun
- ✅ **Complete Pipeline** - Works with prompt generator
- ✅ **Structured Output** - JSON-formatted content

#### **Usage:**
```python
from content_generator import FoodContentPipeline

pipeline = FoodContentPipeline()
result = pipeline.generate_complete_content(
    food_name="Chocolate Cake",
    content_type="recipe",
    style="romantic"
)

print(result["stage_2_content"]["raw_content"])
```

#### **Supported Content Types:**
- **Recipe** - Complete recipes with ingredients and instructions
- **Description** - Marketing and menu descriptions
- **Blog Post** - Engaging blog content
- **Menu Item** - Restaurant menu entries
- **Social Media** - Social media content and captions

## 🎨 Available Features

### **Prompt Generator Styles:**
- `"food blogging"` - Food and culinary content
- `"storytelling"` - Narrative and story-based content
- `"creative"` - Artistic and imaginative content
- `"lifestyle"` - Personal and lifestyle content

### **Prompt Generator Lengths:**
- `"tweet"` - Short, concise prompts (~280 characters)
- `"post"` - Medium-length prompts (~500 characters)
- `"article"` - Long, detailed prompts (~1000 characters)

### **Prompt Generator Themes (Auto-detected):**
- nostalgia, peace, celebration, urban_life, adventure
- creativity, comfort, energy, mystery, growth

### **Content Generator Styles:**
- `"professional"` - Formal, business-like tone
- `"casual"` - Friendly, conversational tone
- `"romantic"` - Elegant, emotional descriptions
- `"elegant"` - Sophisticated, refined language
- `"fun"` - Playful, entertaining content

### **Content Generator Types:**
- `"recipe"` - Complete recipes with ingredients and instructions
- `"description"` - Marketing and menu descriptions
- `"blog_post"` - Engaging blog content
- `"menu_item"` - Restaurant menu entries
- `"social_media"` - Social media content and captions

## 🔧 System Requirements

### **Prompt Generator (Minimum):**
- Python 3.8+
- 2GB RAM
- sentence-transformers, scikit-learn, numpy

### **Prompt Generator (Full Features):**
- Python 3.8+
- 8GB+ RAM
- GPU (recommended)
- transformers, torch, accelerate, peft, bitsandbytes

### **Content Generator:**
- Python 3.8+
- 8GB+ RAM (training), 4GB+ RAM (inference)
- GPU (recommended for training)
- transformers, torch, datasets, accelerate, bitsandbytes

## 📊 Performance

### **Prompt Generator:**
- **Template Mode:** ~0.1 seconds per prompt
- **LLM Mode:** ~2-5 seconds per prompt (depending on hardware)
- **Memory Usage:** ~1-2GB RAM (without LLM), ~4-8GB RAM (with LLM)

### **Content Generator:**
- **Generation Speed:** ~5-10 seconds per content piece
- **Memory Usage:** ~8GB RAM during inference
- **Model Size:** ~4GB (4-bit quantized)
- **Training Time:** ~2-4 hours for 500 samples (with GPU)

## 🚀 Deployment Options

### **1. Direct Python Import**
```python
from portable_model import generate_prompt
# Use anywhere in your Python code
```

### **2. REST API Server**
```bash
python api_server.py
# Access via HTTP requests
```

### **3. Docker Container**
```dockerfile
FROM python:3.11-slim
COPY portable_prompt_generator /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "api_server.py"]
```

### **4. Cloud Platforms**
- AWS Lambda
- Google Cloud Functions
- Azure Functions
- Heroku

## 📞 Support & Documentation

- **`PORTABLE_MODEL_GUIDE.md`** - Complete deployment guide
- **`DEPLOYMENT_GUIDE.md`** - Comprehensive deployment guide
- **`INTEGRATION_SUMMARY.md`** - Integration options
- **`portable_prompt_generator/README.md`** - Portable prompt model documentation
- **`content_generator/README.md`** - Content generator documentation

## 🎉 Ready to Use!

Your intelligent food content generation system is now:
- ✅ **Two-Stage Pipeline** - Prompt generation + Content generation
- ✅ **Portable** - Works on any system
- ✅ **Trainable** - Content generator can be fine-tuned
- ✅ **Complete** - Includes all necessary files and documentation
- ✅ **Reliable** - Always has fallback modes

**Start generating amazing food content!** 🍽️✨ 