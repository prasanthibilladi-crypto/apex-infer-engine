import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine.speculative_decoder import SpeculativeDecoder

def main():
    parser = argparse.ArgumentParser(description="Custom High-Throughput LLM Inference Engine CLI")
    parser.add_argument("--target", type=str, default="gpt2", help="Target model name")
    parser.add_argument("--draft", type=str, default="distilgpt2", help="Draft model name")
    parser.add_argument("--gamma", type=int, default=4, help="Speculative lookahead tokens (K)")
    parser.add_argument("--max_tokens", type=int, default=50, help="Max new tokens to generate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    target_model = AutoModelForCausalLM.from_pretrained(args.target).to(device)
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft).to(device)
    target_model.eval()
    draft_model.eval()

    decoder = SpeculativeDecoder(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        gamma=args.gamma,
        temperature=0.0,
        device=device,
    )

    print("\n" + "=" * 60)
    print(f"✨ Custom Inference Engine Ready (Target: {args.target}, Draft: {args.draft})")
    print("Type your prompt below (or 'exit' to quit):")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input("Prompt > ").strip()
            if not prompt or prompt.lower() in ["exit", "quit"]:
                break

            print("\nGenerating with Speculative Engine...")
            res = decoder.generate(prompt, max_new_tokens=args.max_tokens)
            
            print(f"\n[Result] ({res['tokens_per_sec']:.1f} tok/s | {res['acceptance_rate_pct']:.1f}% acceptance | {res['tokens_per_target_pass']:.2f} tok/target-pass):")
            print("-" * 50)
            print(res["output_text"])
            print("-" * 50 + "\n")

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
