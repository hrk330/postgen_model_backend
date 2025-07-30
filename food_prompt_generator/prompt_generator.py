import warnings
# Suppress transformers warnings
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")

import torch
from transformers.pipelines import pipeline
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import csv
import time
from typing import Optional, Any, List

# Global variable for the generator
generator = None

def test_model_functionality():
    """Test if the loaded model is working properly."""
    global generator
    if generator is None:
        return False
    
    try:
        # Simple test - just check if the generator exists and has required attributes
        if hasattr(generator, 'tokenizer') or hasattr(generator, 'model'):
            return True
        else:
            # For pipeline objects, just return True if they exist
            return True
    except Exception as e:
        print(f"Model test failed: {str(e)}")
        return False

def load_model(model_path: str = "./fine_tuned_llama2_prompt_generator"):
    """
    Load the fine-tuned model with LoRA adapters if available, otherwise fall back to original Llama 2.
    Returns True if successful, False otherwise.
    """
    global generator
    
    try:
        if os.path.exists(model_path):
            print(f"Loading fine-tuned Llama 2 model from {model_path}...")
            try:
                # Try loading with PEFT first - handle configuration issues
                try:
                    config = PeftConfig.from_pretrained(model_path)
                    base_model_name = config.base_model_name_or_path or "meta-llama/Llama-2-7b-chat-hf"
                except Exception as config_error:
                    print(f"PEFT config error: {config_error}")
                    # Try to fix the config file by removing problematic fields
                    import json
                    config_path = os.path.join(model_path, "adapter_config.json")
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r') as f:
                                config_data = json.load(f)
                            # Remove problematic fields
                            if 'corda_config' in config_data:
                                del config_data['corda_config']
                            if 'eva_config' in config_data:
                                del config_data['eva_config']
                            if 'exclude_modules' in config_data:
                                del config_data['exclude_modules']
                            # Write back the cleaned config
                            with open(config_path, 'w') as f:
                                json.dump(config_data, f, indent=2)
                            print("Cleaned adapter_config.json of problematic fields")
                            # Try loading config again
                            config = PeftConfig.from_pretrained(model_path)
                            base_model_name = config.base_model_name_or_path or "meta-llama/Llama-2-7b-chat-hf"
                        except Exception as cleanup_error:
                            print(f"Config cleanup failed: {cleanup_error}")
                            base_model_name = "meta-llama/Llama-2-7b-chat-hf"
                    else:
                        base_model_name = "meta-llama/Llama-2-7b-chat-hf"
                
                # Load base model with GPU optimization for faster generation
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    device_map="cuda",  # Force CUDA for GPU usage
                    torch_dtype="auto",  # Use optimal dtype for GPU
                    low_cpu_mem_usage=True,
                    offload_folder=None,  # Disable disk offloading
                    offload_state_dict=False,  # Keep everything in memory
                    max_memory={0: "8GB"},  # Use more GPU memory for speed
                    load_in_8bit=True  # Use 8-bit quantization for faster loading
                )
                
                # Load LoRA adapters with GPU optimization
                try:
                    model = PeftModel.from_pretrained(
                        base_model,
                        model_path,
                        device_map="cuda"  # Force CUDA for GPU usage
                    )
                except Exception as peft_error:
                    print(f"PEFT loading error: {peft_error}")
                    # Try loading without PEFT
                    model = base_model
                
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # Set pad token if not present
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Create pipeline with better error handling
                try:
                    # Try to merge LoRA weights for better compatibility
                    if hasattr(model, 'merge_and_unload'):
                        print("Merging LoRA weights for better compatibility...")
                        model = model.merge_and_unload()  # type: ignore
                    
                    generator = pipeline(
                        "text-generation",
                        model=model,  # type: ignore
                        tokenizer=tokenizer,
                        device_map="cuda"  # Force CUDA for GPU usage
                    )
                except Exception as pipeline_error:
                    print(f"Pipeline creation failed: {pipeline_error}")
                    # Try with model.model as fallback
                    try:
                        generator = pipeline(
                            "text-generation",
                            model=model.model,  # type: ignore
                            tokenizer=tokenizer,
                            device_map="cuda"  # Force CUDA for GPU usage
                        )
                    except Exception as fallback_error:
                        print(f"Fallback pipeline also failed: {fallback_error}")
                        # Use direct model generation instead of pipeline
                        generator = model
                        generator.tokenizer = tokenizer
                
                print("Successfully loaded fine-tuned model with LoRA adapters.")
                return True
                
            except Exception as e:
                print(f"Error loading fine-tuned model: {str(e)}")
                print("Falling back to original Llama 2...")
                
                # Try loading original model as fallback
                try:
                    generator = pipeline(
                        "text-generation",
                        model="meta-llama/Llama-2-7b-chat-hf",
                        torch_dtype="auto",  # Use optimal dtype for GPU
                        device_map="cuda",  # Force CUDA for GPU usage
                        offload_folder=None
                    )
                    print("Successfully loaded original Llama 2 model.")
                    return True
                except Exception as original_error:
                    print(f"Error loading original model: {str(original_error)}")
                    return False
        else:
            print(f"Fine-tuned model not found at {model_path}, loading original Llama 2...")
            try:
                generator = pipeline(
                    "text-generation",
                    model="meta-llama/Llama-2-7b-chat-hf",
                    torch_dtype="auto",  # Use optimal dtype for GPU
                    device_map="cuda",  # Force CUDA for GPU usage
                    offload_folder=None
                )
                print("Successfully loaded original Llama 2 model.")
                return True
            except Exception as e:
                print(f"Error loading original model: {str(e)}")
                return False
                
    except Exception as e:
        print(f"Critical error in model loading: {str(e)}")
        return False

def template_generate_prompt(keywords, style="food blogging", length="tweet"):
    """
    Generate a prompt using a template when LLM is not available.
    """
    keywords_str = ", ".join(keywords)
    
    if length.lower() == "tweet":
        return f"Generate 2 or 3 tweets on the basis of this prompt: Generate 2-3 different tweets about {keywords_str} in a {style} style."
    else:
        return f"Generate 2 or 3 posts on the basis of this prompt: Generate 2-3 different posts about {keywords_str} in a {style} style."

# Load embedding model once
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Enhanced theme clusters with semantic embeddings
THEME_CLUSTERS = {
    "nostalgia": ["memory", "childhood", "past", "remember", "old", "vintage", "retro", "nostalgic", "yesterday", "history", "journal", "diary", "photo", "album"],
    "peace": ["quiet", "calm", "serene", "peaceful", "tranquil", "meditation", "zen", "mindfulness", "solitude", "silence", "gentle", "soft", "rain", "nature"],
    "celebration": ["party", "festival", "celebration", "joy", "happy", "excited", "cheerful", "birthday", "anniversary", "success", "achievement", "victory", "win"],
    "urban_life": ["city", "urban", "street", "building", "traffic", "crowd", "busy", "fast", "modern", "concrete", "skyscraper", "subway", "taxi", "noise"],
    "adventure": ["explore", "adventure", "journey", "travel", "discover", "mountain", "forest", "ocean", "wild", "daring", "bold", "exploration", "expedition"],
    "creativity": ["art", "creative", "inspiration", "imagination", "design", "paint", "draw", "write", "music", "poetry", "craft", "inventive", "original"],
    "comfort": ["cozy", "warm", "comfort", "home", "family", "love", "hug", "soft", "gentle", "safe", "secure", "familiar", "belonging"],
    "energy": ["power", "energy", "strength", "vitality", "dynamic", "active", "vibrant", "lively", "passionate", "intense", "forceful", "electric"],
    "mystery": ["mystery", "secret", "hidden", "unknown", "curious", "puzzle", "enigma", "strange", "unusual", "mysterious", "dark", "shadow"],
    "growth": ["growth", "learn", "develop", "progress", "improve", "evolve", "change", "transform", "better", "advance", "mature", "bloom"]
}

# Tone definitions based on themes and styles
TONE_MAPPINGS = {
    "nostalgia": "deeply introspective and wistfully romantic, with a gentle longing for days gone by",
    "peace": "serenely contemplative and mindfully present, with a gentle awareness of life's quiet moments", 
    "celebration": "joyfully exuberant and enthusiastically grateful, with a contagious energy that uplifts the spirit",
    "urban_life": "dynamically observant and keenly aware, with a sharp eye for the poetry hidden in city rhythms",
    "adventure": "excitedly curious and boldly courageous, with an insatiable thirst for discovery and wonder",
    "creativity": "inspired and expressively artistic, with a wild imagination that sees beauty in everything",
    "comfort": "warmly nurturing and lovingly gentle, with a soothing presence that feels like coming home",
    "energy": "passionately vibrant and electrically alive, with a raw power that ignites the soul",
    "mystery": "intrigued and contemplatively curious, with a deep fascination for life's hidden wonders",
    "growth": "hopefully determined and courageously evolving, with an unwavering belief in the power of transformation"
}

def extract_theme_from_embeddings(keywords):
    """Extract theme using semantic embeddings and cosine similarity."""
    try:
        # Get embeddings for keywords
        keyword_embeddings = embedding_model.encode(keywords, convert_to_tensor=True)
        
        # Get embeddings for all theme cluster keywords
        all_theme_keywords = []
        theme_names = []
        for theme, keywords_list in THEME_CLUSTERS.items():
            all_theme_keywords.extend(keywords_list)
            theme_names.extend([theme] * len(keywords_list))
        
        theme_embeddings = embedding_model.encode(all_theme_keywords, convert_to_tensor=True)
        
        # Calculate cosine similarity between keyword embeddings and theme embeddings
        # Convert tensors to numpy arrays for sklearn
        try:
            keyword_np = keyword_embeddings.cpu().numpy()  # type: ignore
        except (AttributeError, TypeError):
            keyword_np = keyword_embeddings
            
        try:
            theme_np = theme_embeddings.cpu().numpy()  # type: ignore
        except (AttributeError, TypeError):
            theme_np = theme_embeddings
            
        similarities = cosine_similarity(keyword_np, theme_np)
        
        # Average similarities for each theme
        theme_scores = {}
        for i, theme in enumerate(theme_names):
            if theme not in theme_scores:
                theme_scores[theme] = []
            theme_scores[theme].append(similarities.max(axis=0)[i])
        
        # Get the theme with highest average similarity
        best_theme = max(theme_scores, key=lambda x: sum(theme_scores[x]) / len(theme_scores[x]))
        return best_theme
    except Exception as e:
        print(f"Error in theme extraction: {e}")
        return "creative"  # fallback

def build_scene_summary(keywords, theme):
    """Build a vivid scene summary from keywords and theme."""
    keywords_str = ", ".join(keywords)
    
    scene_templates = {
        "nostalgia": f"diving deep into the treasure chest of memories where {keywords_str} whispers stories of days gone by",
        "peace": f"discovering the sacred sanctuary where {keywords_str} creates moments of pure tranquility and inner calm",
        "celebration": f"dancing in the vibrant symphony of life where {keywords_str} orchestrates moments of pure joy and connection",
        "urban_life": f"navigating the electric pulse of the city where {keywords_str} becomes the heartbeat of modern existence",
        "adventure": f"embarking on an epic quest where {keywords_str} becomes the compass guiding us through uncharted territories of experience",
        "creativity": f"unleashing the wild magic of imagination where {keywords_str} becomes the paintbrush that colors our world with wonder",
        "comfort": f"wrapping ourselves in the warm embrace of familiarity where {keywords_str} becomes the soft blanket that soothes our soul",
        "energy": f"harnessing the raw power of life force where {keywords_str} becomes the lightning that electrifies our existence",
        "mystery": f"peeling back the layers of the unknown where {keywords_str} reveals secrets that dance in the shadows of possibility",
        "growth": f"witnessing the beautiful transformation of the soul where {keywords_str} becomes the catalyst for our evolution into who we're meant to be"
    }
    
    return scene_templates.get(theme, f"exploring the unique combination of {keywords_str}")

def generate_clean_prompt(keywords, style, length, main_theme):
    """Generate an intelligent, scene-based prompt using semantic understanding."""
    keywords_str = ", ".join(keywords)
    
    # Extract theme using semantic embeddings
    detected_theme = extract_theme_from_embeddings(keywords)
    
    # Build scene summary
    scene_summary = build_scene_summary(keywords, detected_theme)
    
    # Get appropriate tone
    tone = TONE_MAPPINGS.get(detected_theme, "engaging and thoughtful")
    
    # Enhanced prompt templates with scene, emotion, and context
    enhanced_prompts = {
        ("food blogging", "tweet"): f"""Generate 2 or 3 tweets on the basis of this prompt: Write 2-3 creative, vivid tweets about {scene_summary}. Build a compelling narrative around {keywords_str} with rich character development and atmospheric details. Use a {tone} tone that creates an immersive storytelling experience. Each tweet should transport readers to the moment, making them feel the textures, aromas, and emotions of the culinary journey.""",
        
        ("food blogging", "post"): f"""Generate 2 or 3 posts on the basis of this prompt: Write 2-3 creative, vivid posts about {scene_summary}. Build a compelling narrative around {keywords_str} with rich character development and atmospheric details. Use a {tone} tone that creates an immersive storytelling experience. Each post should weave together sensory details, cultural significance, personal memories, and emotional depth to create a complete culinary journey that readers can taste, smell, and feel.""",
        
        ("storytelling", "tweet"): f"""Generate 2 or 3 tweets on the basis of this prompt: Write 2-3 creative, vivid tweets about {scene_summary}. Build a compelling narrative around {keywords_str} with rich character development and atmospheric details. Use a {tone} tone that creates an immersive storytelling experience. Each tweet should tell a complete miniature story with a beginning, middle, and end, using magical realism and poetic language to transform ordinary moments into extraordinary tales.""",
        
        ("storytelling", "post"): f"""Generate 2 or 3 posts on the basis of this prompt: Write 2-3 creative, vivid posts about {scene_summary}. Build a compelling narrative around {keywords_str} with rich character development and atmospheric details. Use a {tone} tone that creates an immersive storytelling experience. Each post should read like a scene from a beloved novel, with complex characters, vivid world-building, emotional depth, and narrative tension that keeps readers captivated from beginning to end.""",
        
        ("creative", "tweet"): f"""Generate 2 or 3 tweets on the basis of this prompt: Write 2-3 creative, vivid tweets about {scene_summary}. Build a compelling narrative around {keywords_str} with rich character development and atmospheric details. Use a {tone} tone that creates an immersive storytelling experience. Each tweet should be a work of artistic expression, using innovative metaphors, vivid imagery, and creative language that sparks imagination and inspires readers to see the world through new eyes.""",
        
        ("creative", "post"): f"""Generate 2 or 3 posts on the basis of this prompt: Write 2-3 creative, vivid posts about {scene_summary}. Build a compelling narrative around {keywords_str} with rich character development and atmospheric details. Use a {tone} tone that creates an immersive storytelling experience. Each post should be a creative manifesto that explores artistic inspiration, metaphorical depth, and philosophical meaning, using masterful language to celebrate human creativity and imagination.""",
        
        ("lifestyle", "tweet"): f"""Generate 2 or 3 tweets on the basis of this prompt: Write 2-3 creative, vivid tweets about {scene_summary}. Build a compelling narrative around {keywords_str} with rich character development and atmospheric details. Use a {tone} tone that creates an immersive storytelling experience. Each tweet should inspire positive change and personal growth, offering practical wisdom and encouraging readers to embrace authentic living.""",
        
        ("lifestyle", "post"): f"""Generate 2 or 3 posts on the basis of this prompt: Write 2-3 creative, vivid posts about {scene_summary}. Build a compelling narrative around {keywords_str} with rich character development and atmospheric details. Use a {tone} tone that creates an immersive storytelling experience. Each post should serve as a transformative guide for authentic living, sharing personal transformation stories, practical advice, and life philosophy that empowers readers to create meaningful change."""
    }
    
    return enhanced_prompts.get((style, length), enhanced_prompts[("creative", "tweet")]), detected_theme, tone

def save_prompt_to_csv(keywords, theme, prompt, tone="", csv_path="prompt_generation_log.csv"):
    # Save to CSV with tone
    df = pd.DataFrame([{ 
        'keywords': ', '.join(keywords), 
        'theme': theme, 
        'tone': tone,
        'prompt': prompt 
    }])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

def generate_creative_content(keywords, style, length, main_theme, theme):
    """Generate creative content directly based on keywords and style."""
    keywords_str = ", ".join(keywords)
    
    # Creative templates for different styles and themes
    creative_templates = {
        ("food blogging", "tweet"): [
            f"🍕 Just discovered the most amazing {main_theme} experience! {keywords_str} - pure magic on a plate! ✨ #FoodieLife #Delicious",
            f"🔥 The {main_theme} game is strong with this one! {keywords_str} - every bite tells a story! 🍽️ #FoodAdventure",
            f"💫 When {main_theme} meets {keywords_str} - it's not just food, it's an emotion! 🌟 #FoodLove"
        ],
        ("food blogging", "post"): [
            f"🌟 Today's culinary adventure led me to discover the most incredible {main_theme} experience! The combination of {keywords_str} created a symphony of flavors that danced on my taste buds. Every element tells a story - from the first bite to the last. This is what food dreams are made of! 🍽️✨ #FoodieLife #CulinaryAdventure #Delicious",
            f"🔥 The {main_theme} revelation of the day! Exploring {keywords_str} has opened up a whole new world of flavors. The textures, aromas, and tastes blend together in perfect harmony. This is why I love being a food blogger - discovering these magical combinations! 🍕💫 #FoodAdventure #TasteJourney",
            f"💫 Sometimes the simplest combinations create the most extraordinary experiences. Today's {main_theme} journey with {keywords_str} proved exactly that. It's not just about eating - it's about feeling, experiencing, and falling in love with every moment. Food has the power to transport us to different worlds! 🌟🍽️ #FoodLove #CulinaryMagic"
        ],
        ("storytelling", "tweet"): [
            f"🌙 In the quiet of {main_theme}, where {keywords_str} whispered secrets of forgotten tales... ✨ #StoryTime",
            f"🔥 The {main_theme} held mysteries deeper than the night, where {keywords_str} painted stories in starlight... 🌟 #Adventure",
            f"💫 When {main_theme} and {keywords_str} collide, magic happens in the most unexpected ways... ✨ #Tales"
        ],
        ("storytelling", "post"): [
            f"🌙 There's something magical about {main_theme} that draws you into its embrace. The way {keywords_str} weaves together creates a tapestry of emotions and memories. It's like stepping into a world where every moment holds a story waiting to be told. The quiet whispers of the night, the gentle rustle of forgotten dreams - all coming together in this perfect symphony of life. ✨ #StoryTime #Magic #Adventure",
            f"🔥 The {main_theme} has always held a special place in my heart. When {keywords_str} come together, it's not just a moment - it's an entire universe unfolding before your eyes. The mysteries that lie beneath the surface, the adventures that await around every corner. This is where stories are born, where legends take their first breath. 🌟 #Tales #Mystery #Journey",
            f"💫 Sometimes the most extraordinary stories are hidden in the simplest moments. The {main_theme} teaches us that {keywords_str} can create magic beyond our wildest dreams. It's in these quiet spaces that we find ourselves, that we discover the power of imagination and the beauty of possibility. Every moment is a new chapter waiting to be written. ✨ #Stories #Magic #Life"
        ],
        ("creative", "tweet"): [
            f"🎨 The {main_theme} speaks in colors unseen, where {keywords_str} paint dreams into reality... ✨ #Art #Creativity",
            f"🔥 In the realm of {main_theme}, {keywords_str} become the brushstrokes of imagination... 🌟 #Creative",
            f"💫 When {main_theme} meets {keywords_str}, creativity flows like starlight... ✨ #Inspiration"
        ],
        ("creative", "post"): [
            f"🎨 There's something inherently beautiful about how {main_theme} inspires creativity. The way {keywords_str} blend together creates a canvas of endless possibilities. It's like watching colors dance across the sky, each moment more vibrant than the last. This is where imagination takes flight, where dreams become reality, and where the ordinary transforms into the extraordinary. ✨ #Art #Creativity #Inspiration",
            f"🔥 The {main_theme} has always been my muse, my source of endless inspiration. When {keywords_str} come together, they create a symphony of creative energy that flows through every fiber of my being. It's in these moments that I feel most alive, most connected to the infinite possibilities that surround us. This is the magic of creation. 🌟 #Creative #Art #Magic",
            f"💫 Creativity flows like a river when {main_theme} and {keywords_str} unite. It's like watching the universe paint itself into existence, each stroke more beautiful than the last. The colors, the textures, the emotions - all blending together in perfect harmony. This is where art lives, where inspiration breathes, and where magic happens. ✨ #Creativity #Art #Inspiration"
        ],
        ("lifestyle", "tweet"): [
            f"💪 Embracing the {main_theme} journey with {keywords_str} - every step forward is progress! ✨ #Lifestyle #Motivation",
            f"🔥 The {main_theme} mindset is everything! {keywords_str} - building the life you dream of... 🌟 #Goals",
            f"💫 When {main_theme} meets {keywords_str}, amazing things happen! ✨ #Lifestyle #Success"
        ],
        ("lifestyle", "post"): [
            f"💪 Life is all about embracing the journey, and the {main_theme} path has taught me so much. The combination of {keywords_str} has become my daily inspiration, my motivation to keep pushing forward. Every day is a new opportunity to grow, to learn, and to become the best version of myself. This is what living authentically means - finding what sets your soul on fire and pursuing it with everything you have. ✨ #Lifestyle #Motivation #Growth",
            f"🔥 The {main_theme} mindset has completely transformed my life. When {keywords_str} come together, they create a powerful foundation for success and happiness. It's about building habits that serve you, creating routines that energize you, and surrounding yourself with positivity. Every choice we make shapes our future, and I choose to make choices that align with my dreams and aspirations. 🌟 #Goals #Success #Mindset",
            f"💫 Sometimes the most profound changes come from the simplest shifts in perspective. The {main_theme} approach to life, combined with {keywords_str}, has opened up a world of possibilities I never knew existed. It's about finding balance, creating harmony, and living in alignment with your values. This is the path to true fulfillment and lasting happiness. ✨ #Lifestyle #Balance #Happiness"
        ]
    }
    
    # Get the appropriate templates for this style and length
    templates = creative_templates.get((style, length), creative_templates[("creative", "tweet")])
    
    # Return 2-3 creative outputs
    if length == "tweet":
        return "\n\n".join(templates[:2])  # 2 tweets
    else:
        return "\n\n".join(templates[:3])  # 3 posts

def enhanced_template_generate_prompt(keywords, style="food blogging", length="tweet"):
    # Embedding, clustering, theme extraction, classification
    main_theme = extract_theme_from_embeddings(keywords)
    theme = main_theme # This line is now redundant as theme is extracted directly
    
    # Generate creative content directly instead of just a prompt
    creative_output = generate_creative_content(keywords, style, length, main_theme, theme)
    return creative_output, theme

def llm_generate_prompt(keywords, style="food blogging", length="tweet", quick_mode=False):
    """
    Generate an intelligent, scene-based prompt using semantic understanding.
    Returns (clean_prompt, detected_theme, tone)
    """
    global generator
    start_time = time.time()
    
    # Embedding, clustering, theme extraction, classification
    main_theme = extract_theme_from_embeddings(keywords)
    
    # Generate clean prompt (without few-shot examples)
    clean_prompt, detected_theme, tone = generate_clean_prompt(keywords, style, length, main_theme)
    
    # Quick mode: return clean prompt immediately
    if quick_mode:
        print("Quick mode: Using enhanced prompt generation...")
        save_prompt_to_csv(keywords, detected_theme, clean_prompt, tone)
        elapsed_time = time.time() - start_time
        print(f"Quick generation completed in {elapsed_time:.2f} seconds")
        return clean_prompt, detected_theme, tone
    
    # Try to load model if not already loaded
    if generator is None:
        print("Loading model for intelligent prompt generation...")
        if not load_model():
            print("Model loading failed, using enhanced prompt generation...")
            save_prompt_to_csv(keywords, detected_theme, clean_prompt, tone)
            elapsed_time = time.time() - start_time
            print(f"Enhanced prompt generation completed in {elapsed_time:.2f} seconds")
            return clean_prompt, detected_theme, tone
    
    # Check if generator is available after loading
    if generator is None:
        print("Model not available, using enhanced prompt generation...")
        save_prompt_to_csv(keywords, detected_theme, clean_prompt, tone)
        elapsed_time = time.time() - start_time
        print(f"Enhanced prompt generation completed in {elapsed_time:.2f} seconds")
        return clean_prompt, detected_theme, tone
    
    try:
        print("Using LLM for prompt enhancement...")
        
        # Test if model is working properly with timeout
        try:
            if not test_model_functionality():
                print("Model test failed. Using enhanced prompt fallback...")
                save_prompt_to_csv(keywords, detected_theme, clean_prompt, tone)
                elapsed_time = time.time() - start_time
                print(f"Enhanced prompt generation completed in {elapsed_time:.2f} seconds")
                return clean_prompt, detected_theme, tone
        except Exception as test_error:
            print(f"Model test error: {test_error}. Using enhanced prompt fallback...")
            save_prompt_to_csv(keywords, detected_theme, clean_prompt, tone)
            elapsed_time = time.time() - start_time
            print(f"Enhanced prompt generation completed in {elapsed_time:.2f} seconds")
            return clean_prompt, detected_theme, tone
        
        # Build intelligent prompt for LLM that understands keywords
        keywords_str = ", ".join(keywords)
        input_prompt = f"""You are an expert prompt engineer. Create a highly intelligent and creative prompt based on these keywords: {keywords_str}

The prompt should:
- Deeply understand the semantic relationships between these keywords
- Build a compelling narrative that connects all elements naturally  
- Create rich character development and atmospheric details
- Use the theme: {detected_theme}
- Apply the tone: {tone}
- Focus on {style} style and {length} length

Generate a creative, detailed prompt that goes beyond simple templates and tells a story. Make it immersive and engaging.

Your creative prompt:"""
        pad_token_id = getattr(generator.tokenizer, 'pad_token_id', 50256) or 50256
        
        # Generate with LLM with better error handling
        try:
            result = generator(
                input_prompt,
                max_new_tokens=200,  # Increased for more detailed intelligent prompts
                temperature=0.95,  # Higher for more creativity
                do_sample=True,
                top_p=0.98,  # Higher for more diverse outputs
                top_k=150,  # Increased for better keyword understanding
                num_return_sequences=1,
                pad_token_id=pad_token_id,
                repetition_penalty=1.1,  # Lower to allow more creative repetition
                use_cache=True
            )
            # Handle result safely
            output = ""
            try:
                if result is not None:
                    # Convert to list if it's a generator or iterator
                    result_list = list(result) if hasattr(result, '__iter__') and not isinstance(result, (str, dict, list)) else result
                    
                    if isinstance(result_list, list) and len(result_list) > 0:
                        first_result = result_list[0]
                        if isinstance(first_result, dict) and 'generated_text' in first_result:
                            output = str(first_result['generated_text'])
                        else:
                            output = str(result_list)
                    else:
                        output = str(result_list)
                else:
                    output = ""
            except (TypeError, IndexError, AttributeError):
                output = str(result) if result is not None else ""
            
            # Extract the intelligent prompt from the model output
            if isinstance(output, str):
                # Remove the input prompt from the output
                if "Your creative prompt:" in output:
                    llm_generated_prompt = output.split("Your creative prompt:")[-1].strip()
                elif "You are an expert prompt engineer" in output:
                    # Find the actual generated prompt after the instruction
                    if "Generate a prompt that starts with" in output:
                        llm_generated_prompt = output.split("Generate a prompt that starts with")[-1].strip()
                    else:
                        llm_generated_prompt = output.split("followed by your intelligent, creative prompt:")[-1].strip()
                elif "Analyze these keywords:" in output:
                    llm_generated_prompt = output.split("Generate a prompt that goes beyond simple templates and truly understands the keywords:")[-1].strip()
                elif "Output:" in output:
                    llm_generated_prompt = output.split("Output:")[-1].strip()
                else:
                    llm_generated_prompt = output.strip()
            else:
                llm_generated_prompt = str(output).strip()
            
            # Clean up the output
            llm_generated_prompt = llm_generated_prompt.replace('<s>', '').replace('</s>', '').strip()
            llm_generated_prompt = llm_generated_prompt.lstrip('0123456789. ').strip()
            
            # Add the required prefix to the intelligent prompt
            llm_generated_prompt = f"Generate 2 or 3 {length}s on the basis of this prompt: {llm_generated_prompt}"
            
            # If LLM output is too short or just keywords, use enhanced prompt
            if len(llm_generated_prompt) < 20 or (len(keywords) > 5 and all(keyword.lower() in llm_generated_prompt.lower() for keyword in keywords)):
                print("LLM output too basic, using enhanced prompt...")
                save_prompt_to_csv(keywords, detected_theme, clean_prompt, tone)
                elapsed_time = time.time() - start_time
                print(f"Enhanced prompt generation completed in {elapsed_time:.2f} seconds")
                return clean_prompt, detected_theme, tone
            else:
                save_prompt_to_csv(keywords, detected_theme, llm_generated_prompt, tone)
                elapsed_time = time.time() - start_time
                print(f"LLM prompt generation completed in {elapsed_time:.2f} seconds")
                return llm_generated_prompt, detected_theme, tone
                
        except Exception as generation_error:
            print(f"LLM generation failed: {str(generation_error)}")
            if "input_layernorm.weight" in str(generation_error):
                print("Detected model weight compatibility issue. Using enhanced prompt fallback...")
            else:
                print("Unknown LLM error. Using enhanced prompt fallback...")
            save_prompt_to_csv(keywords, detected_theme, clean_prompt, tone)
            elapsed_time = time.time() - start_time
            print(f"Enhanced prompt generation completed in {elapsed_time:.2f} seconds")
            return clean_prompt, detected_theme, tone
            
    except Exception as e:
        print(f"Error in LLM pipeline: {str(e)}")
        print("Falling back to enhanced prompt generation...")
        save_prompt_to_csv(keywords, detected_theme, clean_prompt, tone)
        elapsed_time = time.time() - start_time
        print(f"Enhanced prompt generation completed in {elapsed_time:.2f} seconds")
        return clean_prompt, detected_theme, tone 

class PromptGenerator:
    """Wrapper class for prompt generation functionality"""
    
    def __init__(self, model_path: str = "./fine_tuned_llama2_prompt_generator"):
        """Initialize the prompt generator"""
        self.model_path = model_path
        self.model_loaded = False
        
        # Load model on initialization
        self._load_model()
    
    def _load_model(self):
        """Load the model"""
        try:
            # Set the model path for the load_model function
            global generator
            if load_model(self.model_path):
                self.model_loaded = True
                print("✅ Prompt Generator model loaded successfully")
            else:
                print("⚠️ Model loading failed, will use fallback methods")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            self.model_loaded = False
    
    def generate_prompt(self, food_type: str, cuisine: str, mood: str, length: str = "medium", additional_context: Optional[str] = None) -> str:
        """
        Generate a food-related prompt based on the given parameters
        
        Args:
            food_type: Type of food (e.g., 'dessert', 'main course')
            cuisine: Cuisine type (e.g., 'italian', 'chinese')
            mood: Mood or atmosphere (e.g., 'romantic', 'casual')
            length: Length of prompt ('short', 'medium', 'long')
            additional_context: Additional context or preferences
            
        Returns:
            Generated prompt string
        """
        # Convert parameters to keywords
        keywords = [food_type, cuisine, mood]
        if additional_context:
            keywords.append(additional_context)
        
        # Map length to style format
        length_mapping = {
            "short": "tweet",
            "medium": "post", 
            "long": "post"
        }
        style_length = length_mapping.get(length, "post")
        
        # Map mood to style
        style_mapping = {
            "romantic": "storytelling",
            "casual": "lifestyle",
            "elegant": "creative",
            "fun": "creative",
            "traditional": "food blogging",
            "modern": "creative"
        }
        style = style_mapping.get(mood.lower(), "food blogging")
        
        try:
            # Try LLM generation first
            if self.model_loaded:
                prompt, theme, tone = llm_generate_prompt(keywords, style, style_length, quick_mode=False)
            else:
                # Fallback to enhanced template generation
                prompt, theme = enhanced_template_generate_prompt(keywords, style, style_length)
            
            return prompt
            
        except Exception as e:
            print(f"Error in prompt generation: {str(e)}")
            # Final fallback to basic template
            return template_generate_prompt(keywords, style, style_length)
    
    def get_model_status(self) -> dict:
        """Get the status of the loaded model"""
        return {
            "model_loaded": self.model_loaded,
            "model_path": self.model_path,
            "model_type": "fine_tuned_llama2_prompt_generator"
        } 