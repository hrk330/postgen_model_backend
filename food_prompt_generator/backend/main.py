"""
Twitter Content Generation Backend API
=====================================

A comprehensive backend system that combines:
1. Keyword extraction from Twitter accounts
2. Intelligent prompt generation based on keywords
3. Content generation using the prompts

Flow: Twitter Account → Keywords → Prompts → Generated Content
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import asyncio
import logging
import os
import sys
from datetime import datetime
import json

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import compatibility modules early to fix huggingface_hub issues
try:
    import huggingface_compatibility
    print("✅ HuggingFace compatibility loaded")
except ImportError:
    print("⚠️  HuggingFace compatibility module not found")

try:
    import transformers_compatibility
    print("✅ Transformers compatibility loaded")
except ImportError:
    print("⚠️  Transformers compatibility module not found")

# Import our models
# Fix import paths for the actual directory structure
from keywords_model.enhanced_keywords_extraction import EnhancedKeywordsExtraction, ExtractionConfig
from prompt_generator import PromptGenerator
from content_generator.content_generator_optimized import OptimizedContentGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Twitter Content Generation API",
    description="AI-powered content generation system using Twitter account analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Pydantic models for API requests/responses
class UserCredentials(BaseModel):
    twitter_username: str = Field(..., description="User's Twitter username (with @)")
    twitter_password: str = Field(..., description="User's Twitter password")
    target_handle: str = Field(..., description="Target Twitter handle to analyze (without @)")

class TwitterAccountRequest(BaseModel):
    user_credentials: UserCredentials = Field(..., description="User's Twitter credentials")
    max_tweets: int = Field(default=100, description="Maximum tweets to analyze")
    
    # Prompt Generator Parameters
    prompt_generation_mode: str = Field(default="fast", description="Prompt generation mode: 'fast' (enhanced templates) or 'full' (LLM + fallback)")
    prompt_style: str = Field(default="food blogging", description="Style for prompt generation: 'food blogging', 'storytelling', 'creative', 'lifestyle'")
    prompt_length: str = Field(default="tweet", description="Length for prompt generation: 'tweet' or 'post'")
    
    # Content Generator Parameters
    content_style: str = Field(default="food blogging", description="Style for content generation")
    content_length: str = Field(default="tweet", description="Length of generated content")
    max_content_length: int = Field(default=200, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.8, description="Temperature for content generation (0.1-1.0)")
    use_trained_model: bool = Field(default=True, description="Whether to use trained model or base model")

class KeywordsRequest(BaseModel):
    keywords: List[str] = Field(..., description="List of keywords to use for generation")
    
    # Prompt Generator Parameters
    prompt_generation_mode: str = Field(default="fast", description="Prompt generation mode: 'fast' (enhanced templates) or 'full' (LLM + fallback)")
    prompt_style: str = Field(default="food blogging", description="Style for prompt generation: 'food blogging', 'storytelling', 'creative', 'lifestyle'")
    prompt_length: str = Field(default="tweet", description="Length for prompt generation: 'tweet' or 'post'")
    
    # Content Generator Parameters
    content_style: str = Field(default="food blogging", description="Style for content generation")
    content_length: str = Field(default="tweet", description="Length of generated content")
    max_content_length: int = Field(default=200, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.8, description="Temperature for content generation (0.1-1.0)")
    use_trained_model: bool = Field(default=True, description="Whether to use trained model or base model")


class KeywordExtractionResponse(BaseModel):
    success: bool
    keywords: List[Dict[str, Any]]
    total_keywords: int
    important_keywords: int
    average_score: float
    message: str

class PromptGenerationResponse(BaseModel):
    success: bool
    prompt: str
    theme: str
    tone: str
    message: str

class ContentGenerationResponse(BaseModel):
    success: bool
    content: List[str]
    total_generated: int
    style: str
    length: str
    message: str

class FullGenerationResponse(BaseModel):
    success: bool
    twitter_handle: str
    keywords: List[Dict[str, Any]]
    prompt: str
    generated_content: List[str]
    processing_time: float
    message: str

# Global instances (initialize once)
keyword_extractor = None
prompt_generator = None
content_generator = None

def initialize_models():
    """Initialize all models globally"""
    global keyword_extractor, prompt_generator, content_generator
    
    try:
        # Get the correct paths relative to backend directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompt_model_path = os.path.join(base_dir, "fine_tuned_llama2_prompt_generator")
        content_model_path = os.path.join(base_dir, "content_generator", "new_trained_content_model")
        
        logger.info(f"🔍 Model paths:")
        logger.info(f"  - Base directory: {base_dir}")
        logger.info(f"  - Prompt model: {prompt_model_path}")
        logger.info(f"  - Content model: {content_model_path}")
        
        # Check if paths exist
        logger.info(f"📁 Path validation:")
        logger.info(f"  - Prompt model exists: {os.path.exists(prompt_model_path)}")
        logger.info(f"  - Content model exists: {os.path.exists(content_model_path)}")
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"🚀 GPU available: {torch.cuda.get_device_name(0)}")
                logger.info(f"  - GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
            else:
                logger.warning("⚠️  GPU not available, using CPU")
        except Exception as e:
            logger.warning(f"⚠️  Could not check GPU: {e}")
        
        # Initialize keyword extraction model
        config = ExtractionConfig(
            max_tweets=100,
            min_keyword_score=0.3,
            top_n_keywords=50,
            cache_enabled=True
        )
        keyword_extractor = EnhancedKeywordsExtraction(config)
        logger.info("✅ Keyword extraction model initialized")
        
        # Initialize prompt generator with trained model path
        logger.info("🔄 Loading prompt generator...")
        prompt_generator = PromptGenerator(model_path=prompt_model_path)
        logger.info("✅ Prompt generator model initialized")
        
        # Initialize content generator with trained model path and GPU
        logger.info("🔄 Loading content generator...")
        content_generator = OptimizedContentGenerator(
            model_path=content_model_path,
            use_gpu=True
        )
        logger.info("✅ Content generator model initialized")
        
        # Log model status
        logger.info("📊 Model Status:")
        logger.info(f"  - Keyword Extractor: ✅ Loaded")
        logger.info(f"  - Prompt Generator: {prompt_generator.get_model_status()}")
        logger.info(f"  - Content Generator: {content_generator.get_model_info()}")
        
    except Exception as e:
        logger.error(f"❌ Error initializing models: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    """Initialize models when the app starts"""
    logger.info("🚀 Starting Twitter Content Generation API...")
    initialize_models()
    logger.info("✅ All models initialized successfully")

# API Routes

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Twitter Content Generation API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "extract_keywords": "/api/v1/extract-keywords",
            "generate_prompt": "/api/v1/generate-prompt", 
            "generate_content": "/api/v1/generate-content",
            "full_generation": "/api/v1/full-generation",
            "generate_from_keywords": "/api/v1/generate-from-keywords",
            "test_models": "/test-models",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "keyword_extractor": keyword_extractor is not None,
            "prompt_generator": prompt_generator is not None,
            "content_generator": content_generator is not None
        }
    }

@app.post("/api/v1/extract-keywords", response_model=KeywordExtractionResponse)
async def extract_keywords(request: TwitterAccountRequest):
    """Extract keywords from a Twitter account using user credentials"""
    try:
        logger.info(f"🔍 Extracting keywords from @{request.user_credentials.target_handle}")
        
        # Check if models are initialized
        if keyword_extractor is None:
            raise HTTPException(status_code=500, detail="Keyword extraction model not initialized")
        
        # Set user credentials for this request
        keyword_extractor.set_credentials(
            username=request.user_credentials.twitter_username,
            password=request.user_credentials.twitter_password,
            target_handle=request.user_credentials.target_handle
        )
        
        # Update config with user preferences
        if hasattr(keyword_extractor, 'config'):
            keyword_extractor.config.max_tweets = request.max_tweets
        
        # Run keyword extraction
        results = keyword_extractor.run_extraction()
        
        # Extract important keywords
        important_keywords = results.get('important_keywords', [])
        
        # Convert to response format
        keywords_data = []
        for _, row in important_keywords.iterrows():
            keywords_data.append({
                "keyword": row['keyword'],
                "score": float(row['score'])
            })
        
        return KeywordExtractionResponse(
            success=True,
            keywords=keywords_data,
            total_keywords=len(results.get('keyword_df', [])),
            important_keywords=len(important_keywords),
            average_score=float(important_keywords['score'].mean()) if len(important_keywords) > 0 else 0.0,
            message=f"Successfully extracted {len(important_keywords)} important keywords from @{request.user_credentials.target_handle}"
        )
        
    except Exception as e:
        logger.error(f"❌ Keyword extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Keyword extraction failed: {str(e)}")

@app.post("/api/v1/generate-prompt", response_model=PromptGenerationResponse)
async def generate_prompt(request: TwitterAccountRequest):
    """Generate a prompt based on extracted keywords"""
    try:
        logger.info(f"🎯 Generating prompt for @{request.user_credentials.target_handle}")
        
        # Check if prompt generator is initialized
        if prompt_generator is None:
            raise HTTPException(status_code=500, detail="Prompt generator model not initialized")
        
        # For this endpoint, we'll use a simple approach without re-scraping
        # You can either pass keywords in the request or use a default approach
        keywords = ["food", "recipe", "cooking", "delicious", "homemade"]
        
        # Generate prompt using the correct method
        prompt = prompt_generator.generate_prompt(
            food_type=keywords[0],
            cuisine=keywords[1],
            mood=request.content_style,
            length=request.content_length,
            additional_context=", ".join(keywords[2:]) if len(keywords) > 2 else None
        )
        
        # For compatibility, extract theme and tone from the prompt
        theme = "food blogging"
        tone = "engaging"
        
        return PromptGenerationResponse(
            success=True,
            prompt=prompt,
            theme=theme,
            tone=tone,
            message=f"Successfully generated prompt for @{request.user_credentials.target_handle}"
        )
        
    except Exception as e:
        logger.error(f"❌ Prompt generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prompt generation failed: {str(e)}")

@app.post("/api/v1/generate-content", response_model=ContentGenerationResponse)
async def generate_content(request: TwitterAccountRequest):
    """Generate content using the prompt"""
    try:
        logger.info(f"📝 Generating content for @{request.user_credentials.target_handle}")
        
        # Check if content generator is initialized
        if content_generator is None:
            raise HTTPException(status_code=500, detail="Content generator model not initialized")
        
        # For this endpoint, we'll use a default prompt without re-scraping
        default_prompt = f"Create engaging {request.content_style} content about food and recipes. Make it {request.content_length} length."
        
        # Generate content using the prompt
        generated_content = content_generator.generate(
            prompt=default_prompt,
            max_length=100,
            temperature=0.8
        )
        
        # Split content into individual tweets/posts
        content_list = [content.strip() for content in generated_content.split('\n\n') if content.strip()]
        
        return ContentGenerationResponse(
            success=True,
            content=content_list,
            total_generated=len(content_list),
            style=request.content_style,
            length=request.content_length,
            message=f"Successfully generated {len(content_list)} content pieces for @{request.user_credentials.target_handle}"
        )
        
    except Exception as e:
        logger.error(f"❌ Content generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Content generation failed: {str(e)}")

@app.post("/api/v1/full-generation", response_model=FullGenerationResponse)
async def full_generation(request: TwitterAccountRequest):
    """Complete workflow: Keywords → Prompt → Content"""
    start_time = datetime.now()
    
    try:
        logger.info(f"🚀 Starting full generation workflow for @{request.user_credentials.target_handle}")
        
        # Check if models are initialized
        if keyword_extractor is None or prompt_generator is None or content_generator is None:
            raise HTTPException(status_code=500, detail="Models not initialized")
        
        # Step 1: Extract keywords (only once)
        logger.info(f"🔍 Step 1: Extracting keywords from @{request.user_credentials.target_handle}")
        
        # Set user credentials for this request
        keyword_extractor.set_credentials(
            username=request.user_credentials.twitter_username,
            password=request.user_credentials.twitter_password,
            target_handle=request.user_credentials.target_handle
        )
        
        # Update config with user preferences
        if hasattr(keyword_extractor, 'config'):
            keyword_extractor.config.max_tweets = request.max_tweets
        
        # Run keyword extraction (this scrapes Twitter once)
        results = keyword_extractor.run_extraction()
        
        # Extract important keywords
        important_keywords = results.get('important_keywords', [])
        
        # Convert to response format
        keywords_data = []
        for _, row in important_keywords.iterrows():
            keywords_data.append({
                "keyword": row['keyword'],
                "score": float(row['score'])
            })
        
        logger.info(f"✅ Extracted {len(keywords_data)} keywords")
        
        # Step 2: Generate prompt using extracted keywords
        logger.info(f"🎯 Step 2: Generating prompt")
        
        # Extract keywords for prompt generation
        keywords = [kw['keyword'] for kw in keywords_data[:10]]  # Top 10 keywords
        
        # Generate prompt using the correct method with user-specified parameters
        prompt = prompt_generator.generate_prompt(
            food_type=keywords[0] if keywords else "food",
            cuisine=keywords[1] if len(keywords) > 1 else "general",
            mood=request.prompt_style,  # Use prompt_style instead of content_style
            length=request.prompt_length,  # Use prompt_length instead of content_length
            additional_context=", ".join(keywords[2:]) if len(keywords) > 2 else None
        )
        
        # For compatibility, extract theme and tone from the prompt
        theme = "food blogging"
        tone = "engaging"
        
        logger.info(f"✅ Generated prompt using trained model")
        
        # Step 3: Generate content using the prompt
        logger.info(f"📝 Step 3: Generating content")
        
        # Generate content using the prompt with user-specified parameters
        generated_content = content_generator.generate(
            prompt=prompt,
            max_length=request.max_content_length,  # Use user-specified max length
            temperature=request.temperature  # Use user-specified temperature
        )
        
        # Split content into individual tweets/posts
        content_list = [content.strip() for content in generated_content.split('\n\n') if content.strip()]
        
        logger.info(f"✅ Generated {len(content_list)} content pieces")
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return FullGenerationResponse(
            success=True,
            twitter_handle=request.user_credentials.target_handle,
            keywords=keywords_data,
            prompt=prompt,
            generated_content=content_list,
            processing_time=processing_time,
            message=f"Successfully completed full generation workflow for @{request.user_credentials.target_handle}"
        )
        
    except Exception as e:
        logger.error(f"❌ Full generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Full generation failed: {str(e)}")

@app.post("/api/v1/generate-from-keywords", response_model=FullGenerationResponse)
async def generate_from_keywords(request: KeywordsRequest):
    """Generate content from provided keywords (no Twitter scraping)"""
    start_time = datetime.now()
    
    try:
        logger.info(f"🚀 Starting generation from keywords: {request.keywords[:5]}...")
        
        # Check if models are initialized
        if prompt_generator is None or content_generator is None:
            raise HTTPException(status_code=500, detail="Models not initialized")
        
        # Step 1: Convert keywords to the format expected by the system
        keywords_data = []
        for i, keyword in enumerate(request.keywords):
            keywords_data.append({
                "keyword": keyword,
                "score": 0.8 - (i * 0.05)  # Decreasing score for ranking
            })
        
        logger.info(f"✅ Processed {len(keywords_data)} keywords")
        
        # Step 2: Generate prompt using keywords
        logger.info(f"🎯 Step 2: Generating prompt")
        
        # Use the first few keywords for prompt generation
        keywords = request.keywords[:10]
        
        # Generate prompt using the correct method with user-specified parameters
        prompt = prompt_generator.generate_prompt(
            food_type=keywords[0] if keywords else "food",
            cuisine=keywords[1] if len(keywords) > 1 else "general",
            mood=request.prompt_style,  # Use prompt_style instead of content_style
            length=request.prompt_length,  # Use prompt_length instead of content_length
            additional_context=", ".join(keywords[2:]) if len(keywords) > 2 else None
        )
        
        logger.info(f"✅ Generated prompt")
        
        # Step 3: Generate content using the prompt
        logger.info(f"📝 Step 3: Generating content")
        
        # Generate content using the prompt with user-specified parameters
        generated_content = content_generator.generate(
            prompt=prompt,
            max_length=request.max_content_length,  # Use user-specified max length
            temperature=request.temperature  # Use user-specified temperature
        )
        
        # Split content into individual tweets/posts
        content_list = [content.strip() for content in generated_content.split('\n\n') if content.strip()]
        
        logger.info(f"✅ Generated {len(content_list)} content pieces")
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return FullGenerationResponse(
            success=True,
            twitter_handle="provided_keywords",
            keywords=keywords_data,
            prompt=prompt,
            generated_content=content_list,
            processing_time=processing_time,
            message=f"Successfully generated content from {len(request.keywords)} provided keywords"
        )
        
    except Exception as e:
        logger.error(f"❌ Generation from keywords failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation from keywords failed: {str(e)}")

# Background task for long-running operations
@app.post("/api/v1/async-generation")
async def async_generation(request: TwitterAccountRequest, background_tasks: BackgroundTasks):
    """Async generation with background processing"""
    try:
        # Start background task
        background_tasks.add_task(process_async_generation, request)
        
        return {
            "success": True,
            "message": f"Started async generation for @{request.user_credentials.target_handle}",
            "job_id": f"gen_{request.user_credentials.target_handle}_{datetime.now().timestamp()}"
        }
        
    except Exception as e:
        logger.error(f"❌ Async generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Async generation failed: {str(e)}")

async def process_async_generation(request: TwitterAccountRequest):
    """Background task for processing async generation"""
    try:
        # This would typically save results to a database
        # For now, we'll just log the completion
        logger.info(f"🔄 Processing async generation for @{request.user_credentials.target_handle}")
        
        # Simulate processing time
        await asyncio.sleep(5)
        
        logger.info(f"✅ Async generation completed for @{request.user_credentials.target_handle}")
        
    except Exception as e:
        logger.error(f"❌ Async generation processing failed: {e}")

@app.get("/test-models")
async def test_models():
    """Test endpoint to verify model functionality"""
    try:
        results = {
            "keyword_extractor": keyword_extractor is not None,
            "prompt_generator": {
                "loaded": prompt_generator is not None,
                "model_status": prompt_generator.get_model_status() if prompt_generator else None
            },
            "content_generator": {
                "loaded": content_generator is not None,
                "model_info": content_generator.get_model_info() if content_generator else None
            }
        }
        
        # Test prompt generation
        if prompt_generator:
            try:
                test_prompt = prompt_generator.generate_prompt(
                    food_type="pasta",
                    cuisine="italian", 
                    mood="food blogging",
                    length="tweet"
                )
                results["prompt_test"] = {
                    "success": True,
                    "prompt_length": len(test_prompt),
                    "prompt_preview": test_prompt[:100] + "..." if len(test_prompt) > 100 else test_prompt
                }
            except Exception as e:
                results["prompt_test"] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Test content generation
        if content_generator:
            try:
                test_content = content_generator.generate(
                    prompt="Write a short tweet about delicious pasta",
                    max_length=50,
                    temperature=0.8
                )
                results["content_test"] = {
                    "success": True,
                    "content_length": len(test_content),
                    "content_preview": test_content[:100] + "..." if len(test_content) > 100 else test_content
                }
            except Exception as e:
                results["content_test"] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 