from prompt_generator import llm_generate_prompt

def main():
    try:
        # Get keywords with validation
        keywords_input = input("Enter keywords (comma separated): ").strip()
        if not keywords_input:
            print("Error: No keywords provided.")
            return
            
        keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
        if not keywords:
            print("Error: No valid keywords found.")
            return
            
        print(f"Processing keywords: {keywords}")
        
        # Get generation mode
        print("Generation mode:")
        print("1. Fast mode (enhanced templates only)")
        print("2. Full mode (LLM + fallback)")
        mode_choice = input("Enter choice (1-2, default: 1): ").strip()
        quick_mode = (mode_choice != '2')
        
        # Get style with simple options
        print("Style options:")
        print("1. Food blogging (default)")
        print("2. Storytelling")
        print("3. Creative")
        print("4. Lifestyle")
        style_choice = input("Enter choice (1-4, default: 1): ").strip()
        if style_choice == '2':
            style = "storytelling"
        elif style_choice == '3':
            style = "creative"
        elif style_choice == '4':
            style = "lifestyle"
        else:
            style = "food blogging"
            
        # Get length with simple options
        print("Length options:")
        print("1. Tweet (default)")
        print("2. Post")
        length_choice = input("Enter choice (1-2, default: 1): ").strip()
        if length_choice == '2':
            length = "post"
        else:
            length = "tweet"
            
        print(f"Style: {style}, Length: {length}")
        print(f"Mode: {'Fast' if quick_mode else 'Full'}")
        
        print("Generating intelligent prompt...")
        prompt, theme, tone = llm_generate_prompt(keywords, style, length, quick_mode)
        print(f"\n🎯 Detected Theme: {theme}")
        print(f"🎨 Suggested Tone: {tone}")
        print(f"\n✨ Enhanced AI Prompt:\n")
        print(prompt)
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 