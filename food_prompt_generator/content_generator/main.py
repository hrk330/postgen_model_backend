"""
Content Generator CLI with Training Support - Ultra-Optimized for Lowest Latency

A command-line interface for the Llama 2 content generator with separate training capabilities.
This CLI can use both pre-trained and custom-trained models with ultra-optimized caching.
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path

# Fix import for direct script execution
try:
    from .content_generator_optimized import OptimizedContentGenerator, generate_content, get_global_generator
    from .train_new_model import train_new_content_model
except ImportError:
    # When running script directly
    from content_generator_optimized import OptimizedContentGenerator, generate_content, get_global_generator
    from train_new_model import train_new_content_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simple_generate(args):
    """Generate content from a simple prompt using Llama 2 with menu system for multiple generations."""
    try:
        start_time = time.time()
        print(f"🚀 Loading Content Generator...")
        
        # Prompt for model selection if not provided
        if not args.model_path:
            print("\nWhich model do you want to use?")
            print("  1) Base Llama 2 (meta-llama/Llama-2-7b-chat-hf)")
            print("  2) Your trained model (./trained_llama2_qlora_full_dataset)")
            while True:
                choice = input("Select model [1/2]: ").strip()
                if choice == '1':
                    args.model_path = "meta-llama/Llama-2-7b-chat-hf"
                    break
                elif choice == '2':
                    args.model_path = "./trained_llama2_qlora_full_dataset"
                    break
                else:
                    print("Please enter 1 or 2.")
        
        # Get global generator instance for maximum caching efficiency
        generator = get_global_generator(args.model_path, not args.cpu_only)
        
        if not generator.model:
            print("❌ No model loaded. Please check your internet connection for downloading pre-trained models.")
            return
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        print(f"📁 Model: {args.model_path or 'Default Llama 2'}")
        
        # Check if prompt was provided or we should enter interactive mode
        if args.prompt:
            # Single generation mode
            print(f"\n🚀 Generating content for prompt: {args.prompt}")
            print(f"📊 Settings: Max length={args.max_length}, Temperature={args.temperature}")
            
            gen_start_time = time.time()
            content = generator.generate(
                prompt=args.prompt,
                max_length=args.max_length,
                temperature=args.temperature
            )
            gen_time = time.time() - gen_start_time
            
            print("\n" + "="*60)
            print("GENERATED CONTENT")
            print("="*60)
            print(f"⏱️  Generation time: {gen_time:.2f} seconds")
            print(f"📝 Response length: {len(content)} characters")
            print("\nResponse:")
            print("-" * 40)
            print(content)
            print("="*60)
            return
        
        # Interactive menu system for multiple generations
        print("\n" + "="*60)
        print("🤖 CONTENT GENERATOR - MENU SYSTEM")
        print("="*60)
        print("Commands:")
        print("  - Type your prompt to generate content")
        print("  - 'help' - Show this help message")
        print("  - 'settings' - Change generation settings")
        print("  - 'info' - Show model information")
        print("  - 'stats' - Show generation statistics")
        print("  - 'quit' or 'exit' - Exit the program")
        print("="*60)
        
        # Track statistics
        generation_count = 0
        total_generation_time = 0
        current_max_length = args.max_length
        current_temperature = args.temperature
        
        while True:
            try:
                # Get user input
                user_input = input("\n📝 Enter your prompt (or command): ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    if generation_count > 0:
                        avg_time = total_generation_time / generation_count
                        print(f"\n📊 Session Summary:")
                        print(f"   Total generations: {generation_count}")
                        print(f"   Total generation time: {total_generation_time:.2f} seconds")
                        print(f"   Average generation time: {avg_time:.2f} seconds")
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("\n📖 Available Commands:")
                    print("  - Type any prompt to generate content")
                    print("  - 'help' - Show this help message")
                    print("  - 'settings' - Change max_length and temperature")
                    print("  - 'info' - Show model information")
                    print("  - 'stats' - Show generation statistics")
                    print("  - 'quit' or 'exit' - Exit the program")
                    print(f"\n📊 Current Settings:")
                    print(f"   Max length: {current_max_length}")
                    print(f"   Temperature: {current_temperature}")
                    continue
                
                if user_input.lower() == 'settings':
                    print(f"\n⚙️  Current Settings:")
                    print(f"   Max length: {current_max_length}")
                    print(f"   Temperature: {current_temperature}")
                    
                    try:
                        new_max_length = input(f"Enter new max length (current: {current_max_length}): ").strip()
                        if new_max_length:
                            current_max_length = int(new_max_length)
                        
                        new_temperature = input(f"Enter new temperature (current: {current_temperature}): ").strip()
                        if new_temperature:
                            current_temperature = float(new_temperature)
                        
                        print(f"✅ Settings updated: Max length={current_max_length}, Temperature={current_temperature}")
                    except ValueError:
                        print("❌ Invalid input. Settings unchanged.")
                    continue
                
                if user_input.lower() == 'info':
                    info = generator.get_model_info()
                    print("\n📊 Model Information:")
                    for key, value in info.items():
                        print(f"  {key}: {value}")
                    continue
                
                if user_input.lower() == 'stats':
                    if generation_count > 0:
                        avg_time = total_generation_time / generation_count
                        print(f"\n📊 Generation Statistics:")
                        print(f"  Total generations: {generation_count}")
                        print(f"  Total generation time: {total_generation_time:.2f} seconds")
                        print(f"  Average generation time: {avg_time:.2f} seconds")
                        print(f"  Current settings: Max length={current_max_length}, Temperature={current_temperature}")
                    else:
                        print("📊 No generations yet in this session")
                    continue
                
                # Generate content
                print(f"\n🚀 Generating content...")
                print(f"📊 Settings: Max length={current_max_length}, Temperature={current_temperature}")
                
                gen_start_time = time.time()
                content = generator.generate(
                    prompt=user_input,
                    max_length=current_max_length,
                    temperature=current_temperature
                )
                gen_time = time.time() - gen_start_time
                
                # Update statistics
                generation_count += 1
                total_generation_time += gen_time
                
                # Display result
                print("\n" + "="*60)
                print("GENERATED CONTENT")
                print("="*60)
                print(f"⏱️  Generation time: {gen_time:.2f} seconds")
                print(f"📝 Response length: {len(content)} characters")
                print("\nResponse:")
                print("-" * 40)
                print(content)
                print("="*60)
                
                # Ask if user wants to continue
                print(f"\n💡 Tip: Type another prompt to generate more content, or 'quit' to exit")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                print("💡 Try again with a different prompt")
        
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        print(f"Error: {str(e)}")


def interactive_mode(args):
    """Interactive chat mode using Llama 2 with ultra-optimized caching."""
    try:
        # Get global generator instance for maximum caching efficiency
        start_time = time.time()
        print("🤖 Loading Llama 2 model...")
        generator = get_global_generator(args.model_path, not args.cpu_only)
        load_time = time.time() - start_time
        
        if not generator.model:
            print("❌ No model loaded. Please check your internet connection for downloading pre-trained models.")
            print("Note: Llama 2 requires more memory but provides excellent quality.")
            return
        
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        print(f"📁 Model path: {args.model_path or 'Default Llama 2'}")
        print("\n🤖 Content Generator - Interactive Mode (Llama 2)")
        print("Type 'quit' to exit, 'help' for commands, 'clear' to clear cache")
        print("="*50)
        
        conversation_history = []
        total_generation_time = 0
        generation_count = 0
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    if generation_count > 0:
                        avg_time = total_generation_time / generation_count
                        print(f"\n📊 Session Stats:")
                        print(f"   Total generations: {generation_count}")
                        print(f"   Total generation time: {total_generation_time:.2f} seconds")
                        print(f"   Average generation time: {avg_time:.2f} seconds")
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("\n📖 Available commands:")
                    print("  - Type any prompt to generate content")
                    print("  - 'quit' or 'exit' to leave")
                    print("  - 'clear' to clear conversation history")
                    print("  - 'cache' to clear model cache (frees memory)")
                    print("  - 'info' to see model information")
                    print("  - 'stats' to see session statistics")
                    continue
                
                if user_input.lower() == 'clear':
                    conversation_history = []
                    print("🗑️  Conversation history cleared")
                    continue
                
                if user_input.lower() == 'cache':
                    generator.clear_cache()
                    print("🗑️  Model cache cleared (memory freed)")
                    continue
                
                if user_input.lower() == 'info':
                    info = generator.get_model_info()
                    print("\n📊 Model Information:")
                    for key, value in info.items():
                        print(f"  {key}: {value}")
                    continue
                
                if user_input.lower() == 'stats':
                    if generation_count > 0:
                        avg_time = total_generation_time / generation_count
                        print(f"\n📊 Session Statistics:")
                        print(f"  Total generations: {generation_count}")
                        print(f"  Total generation time: {total_generation_time:.2f} seconds")
                        print(f"  Average generation time: {avg_time:.2f} seconds")
                    else:
                        print("📊 No generations yet in this session")
                    continue
                
                if not user_input:
                    continue
                
                # Add user message to history
                conversation_history.append({"role": "user", "content": user_input})
                
                # Generate response with timing
                print("🤖 Generating...")
                gen_start_time = time.time()
                response = generator.chat(
                    conversation_history,
                    max_length=args.max_length,
                    temperature=args.temperature
                )
                gen_time = time.time() - gen_start_time
                
                # Update statistics
                total_generation_time += gen_time
                generation_count += 1
                
                # Add assistant response to history
                conversation_history.append({"role": "assistant", "content": response})
                
                print(f"🤖 Assistant ({gen_time:.2f}s): {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error in interactive mode: {str(e)}")


def train_model(args):
    """Train Llama 2 on custom dataset."""
    try:
        print("🚀 Starting Llama 2 training on custom dataset...")
        print(f"📁 Data file: {args.data_file}")
        print(f"📁 Output directory: {args.output_dir}")
        print(f"📊 Training parameters:")
        print(f"   - Epochs: {args.epochs}")
        print(f"   - Batch size: {args.batch_size}")
        print(f"   - Learning rate: {args.learning_rate}")
        print("="*50)
        
        # Check if data file exists
        if not Path(args.data_file).exists():
            print(f"❌ Data file not found: {args.data_file}")
            print("Please create a CSV file with 'prompt' and 'response' columns")
            return
        
        # Validate training data first
        print("🔍 Validating training data...")
        import pandas as pd
        try:
            df = pd.read_csv(args.data_file)
            required_columns = ["prompt", "response"]
            
            if not all(col in df.columns for col in required_columns):
                print("❌ CSV file must contain 'prompt' and 'response' columns")
                return
            
            df_clean = df.dropna(subset=required_columns)
            print(f"✅ Training data validated: {len(df_clean)} valid examples")
            
        except Exception as e:
            print(f"❌ Error reading CSV file: {str(e)}")
            return
        
        # Start training
        try:
            train_new_content_model(
                csv_file=args.data_file,
                output_dir=args.output_dir,
                model_name=args.base_model,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            print(f"\n✅ Training completed successfully!")
            print(f"📁 Trained model saved to: {args.output_dir}")
            print(f"\nTo use your trained model:")
            print(f"python main.py generate --prompt 'Your prompt' --model-path {args.output_dir}")
            print(f"python main.py chat --model-path {args.output_dir}")
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        print(f"Error: {str(e)}")


def validate_data(args):
    """Validate training data format and content."""
    try:
        print(f"🔍 Validating training data: {args.data_file}")
        
        if not Path(args.data_file).exists():
            print(f"❌ Data file not found: {args.data_file}")
            return
        
        # Simple validation for CSV files
        import pandas as pd
        try:
            df = pd.read_csv(args.data_file)
            required_columns = ["prompt", "response"]
            
            if not all(col in df.columns for col in required_columns):
                print("❌ CSV file must contain 'prompt' and 'response' columns")
                return
            
            # Remove rows with missing data
            df_clean = df.dropna(subset=required_columns)
            
            print("✅ Training data is valid!")
            print(f"📊 Statistics:")
            print(f"   - Total examples: {len(df)}")
            print(f"   - Valid examples: {len(df_clean)}")
            print(f"   - Invalid examples: {len(df) - len(df_clean)}")
            
        except Exception as e:
            print(f"❌ Error reading CSV file: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        print(f"Error: {str(e)}")


def create_sample_dataset(args):
    """Create a sample training dataset."""
    try:
        output_file = args.output_file
        
        # Create sample training data
        import pandas as pd
        
        sample_data = [
            {"prompt": "Write a short story about a robot", "response": "Once upon a time, there was a friendly robot named Robo who loved to help people."},
            {"prompt": "Explain quantum physics", "response": "Quantum physics is a branch of physics that describes the behavior of matter and energy at the atomic and subatomic level."},
            {"prompt": "Give me a recipe for chocolate cake", "response": "Mix flour, sugar, cocoa, eggs, milk, and oil. Bake at 350°F for 30 minutes."},
            {"prompt": "What is machine learning?", "response": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."},
            {"prompt": "Tell me a joke", "response": "Why don't scientists trust atoms? Because they make up everything!"}
        ]
        
        df = pd.DataFrame(sample_data)
        df.to_csv(output_file, index=False)
        
        print(f"✅ Sample dataset created: {output_file}")
        print(f"📊 Contains {len(sample_data)} training examples")
        print(f"\nTo validate the dataset:")
        print(f"python main.py validate --data-file {output_file}")
        print(f"\nTo train your model:")
        print(f"python main.py train --data-file {output_file} --output-dir ./my_trained_model")
        
    except Exception as e:
        logger.error(f"Error creating sample dataset: {str(e)}")
        print(f"Error: {str(e)}")


def test_model(args):
    """Test the Llama 2 content generation model with caching."""
    try:
        # Get global generator instance for maximum caching efficiency
        start_time = time.time()
        print("🤖 Loading Llama 2 model for testing...")
        generator = get_global_generator(args.model_path, not args.cpu_only)
        load_time = time.time() - start_time
        
        if not generator.model:
            logger.error("No model loaded. Please check your internet connection for downloading pre-trained models.")
            return
        
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        print(f"📁 Model path: {args.model_path or 'Default Llama 2'}")
        
        # Test the model
        test_start_time = time.time()
        if generator.test_model():
            test_time = time.time() - test_start_time
            print(f"✅ Llama 2 model test successful in {test_time:.2f} seconds!")
            
            # Get model info
            info = generator.get_model_info()
            print("\n📊 Model Information:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print("❌ Llama 2 model test failed!")
            
    except Exception as e:
        logger.error(f"Error testing model: {str(e)}")


def main():
    """Main function with comprehensive command line interface."""
    parser = argparse.ArgumentParser(
        description="Llama 2 Content Generator with Training Support - High Quality AI Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate content from prompt (uses Llama 2)
  python main.py generate --prompt "Hello, how are you?"

  # Interactive generation mode (no prompt needed)
  python main.py generate

  # Use a trained model
  python main.py generate --prompt "Your prompt" --model-path ./my_trained_model

  # Interactive chat mode (uses Llama 2)
  python main.py chat

  # Interactive chat with trained model
  python main.py chat --model-path ./my_trained_model

  # Test Llama 2 model
  python main.py test

  # Create sample training dataset
  python main.py create-dataset --output-file training_data.json

  # Validate training data
  python main.py validate --data-file training_data.json

  # Train on custom dataset
  python main.py train --data-file training_data.json --output-dir ./trained_model

  # Generate with custom settings
  python main.py generate --prompt "Tell me a joke" --max-length 300 --temperature 0.8

  # Use CPU mode (if you have memory issues)
  python main.py generate --prompt "Hello" --cpu-only

Note: Llama 2 provides excellent quality and is great for fine-tuning on custom datasets.
After training, use --model-path to specify your trained model for generation.
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate content from prompt using Llama 2 (interactive menu system)')
    gen_parser.add_argument('--prompt', help='Your prompt (optional - will enter interactive mode if not provided)')
    gen_parser.add_argument('--model-path', help='Path to model or Hugging Face model name (default: meta-llama/Llama-2-7b-chat-hf)')
    gen_parser.add_argument('--max-length', type=int, default=200, help='Maximum response length')
    gen_parser.add_argument('--temperature', type=float, default=0.7, help='Creativity level (0.1-1.0)')
    gen_parser.add_argument('--cpu-only', action='store_true', help='Use CPU only (recommended if you have memory issues)')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat mode using Llama 2')
    chat_parser.add_argument('--model-path', help='Path to model or Hugging Face model name (default: meta-llama/Llama-2-7b-chat-hf)')
    chat_parser.add_argument('--max-length', type=int, default=200, help='Maximum response length')
    chat_parser.add_argument('--temperature', type=float, default=0.7, help='Creativity level (0.1-1.0)')
    chat_parser.add_argument('--cpu-only', action='store_true', help='Use CPU only (recommended if you have memory issues)')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train Llama 2 on custom dataset')
    train_parser.add_argument('--data-file', required=True, help='Path to JSON training data file')
    train_parser.add_argument('--output-dir', default='./trained_model', help='Output directory for trained model')
    train_parser.add_argument('--base-model', default='meta-llama/Llama-2-7b-chat-hf', help='Base model to fine-tune')
    train_parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')
    train_parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate training data format and content')
    validate_parser.add_argument('--data-file', required=True, help='Path to JSON training data file')
    
    # Create dataset command
    dataset_parser = subparsers.add_parser('create-dataset', help='Create a sample training dataset')
    dataset_parser.add_argument('--output-file', required=True, help='Output JSON file for sample dataset')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test Llama 2 content generation model')
    test_parser.add_argument('--model-path', help='Path to model or Hugging Face model name (default: meta-llama/Llama-2-7b-chat-hf)')
    test_parser.add_argument('--cpu-only', action='store_true', help='Use CPU only (recommended if you have memory issues)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute the appropriate function
    if args.command == 'generate':
        simple_generate(args)
    elif args.command == 'chat':
        interactive_mode(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'validate':
        validate_data(args)
    elif args.command == 'create-dataset':
        create_sample_dataset(args)
    elif args.command == 'test':
        test_model(args)


if __name__ == "__main__":
    main()