import sys
import time
import torch
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine.block_allocator import BlockAllocator
from engine.speculative_decoder import SpeculativeDecoder

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def run_standard_autoregressive(model, tokenizer, prompt: str, max_new_tokens: int, device):
    """Standard HuggingFace autoregressive generation baseline."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    start_time = time.time()
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    total_time = time.time() - start_time
    tokens_generated = output.shape[1] - input_ids.shape[1]
    tokens_per_sec = tokens_generated / total_time if total_time > 0 else 0.0
    text = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    
    return {
        "text": text,
        "tokens": tokens_generated,
        "time": total_time,
        "tok_per_sec": tokens_per_sec,
    }

def main():
    print("=" * 70)
    print("[*] LLM INFERENCE ENGINE BENCHMARK: BASELINE vs. SPECULATIVE + PAGED CACHE")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # For lightweight, universal testing on any machine:
    # Target Model: gpt2 (124M) or gpt2-medium
    # Draft Model: distilgpt2 (82M)
    target_model_name = "gpt2"
    draft_model_name = "distilgpt2"

    print(f"\n[1/3] Loading Target Model: '{target_model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    target_model = AutoModelForCausalLM.from_pretrained(target_model_name).to(device)
    target_model.eval()

    print(f"[2/3] Loading Draft Model: '{draft_model_name}'...")
    draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name).to(device)
    draft_model.eval()

    prompt = "The key difference between machine learning algorithms and deep neural networks is"
    max_new_tokens = 40
    print(f"\nPrompt: \"{prompt}\"")
    print(f"Max New Tokens: {max_new_tokens}\n")

    # -------------------------------------------------------------
    # 1. Run Standard Baseline
    # -------------------------------------------------------------
    print("--> Running Standard Baseline (HuggingFace Autoregressive)...")
    baseline_res = run_standard_autoregressive(target_model, tokenizer, prompt, max_new_tokens, device)
    print(f"    Done in {baseline_res['time']:.2f}s ({baseline_res['tok_per_sec']:.1f} tokens/sec)")

    # -------------------------------------------------------------
    # 2. Run Custom Speculative Engine
    # -------------------------------------------------------------
    print("--> Running Custom Speculative Decoding Engine (K=4 lookahead)...")
    spec_decoder = SpeculativeDecoder(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        gamma=4,
        temperature=0.0,
        device=device,
    )
    spec_res = spec_decoder.generate(prompt, max_new_tokens=max_new_tokens)
    print(f"    Done in {spec_res['total_time_sec']:.2f}s ({spec_res['tokens_per_sec']:.1f} tokens/sec)")

    # -------------------------------------------------------------
    # 3. Simulate Paged KV-Cache Memory Efficiency
    # -------------------------------------------------------------
    print("\n--> Evaluating Paged KV-Cache Virtual Memory Allocator...")
    allocator = BlockAllocator(
        num_layers=12,
        num_heads=12,
        head_dim=64,
        total_num_blocks=256,
        block_size=16,
        device=device,
    )
    # Simulate 8 concurrent requests of varied lengths
    req_lengths = [25, 43, 110, 85, 12, 60, 95, 33]
    for i, seq_len in enumerate(req_lengths):
        allocator.allocate_sequence(f"req_{i}", seq_len)
    
    mem_stats = allocator.get_memory_stats()

    # -------------------------------------------------------------
    # Comparison Summary Table
    # -------------------------------------------------------------
    speedup = spec_res["tokens_per_sec"] / baseline_res["tok_per_sec"] if baseline_res["tok_per_sec"] > 0 else 1.0

    table_data = [
        ["Throughput (tokens/sec)", f"{baseline_res['tok_per_sec']:.2f} tok/s", f"{spec_res['tokens_per_sec']:.2f} tok/s", f"{speedup:.2f}x"],
        ["Target Forward Passes", f"{baseline_res['tokens']}", f"{spec_res['target_forward_passes']}", f"Reduced by {(1 - spec_res['target_forward_passes']/baseline_res['tokens'])*100:.1f}%"],
        ["Tokens Generated / Pass", "1.0 tok/pass", f"{spec_res['tokens_per_target_pass']:.2f} tok/pass", f"{spec_res['tokens_per_target_pass']:.2f}x"],
        ["Speculative Acceptance Rate", "N/A", f"{spec_res['acceptance_rate_pct']:.1f}%", "Acceptance Rate"],
        ["KV Memory Fragmentation", "50% - 70% (Contiguous waste)", f"{mem_stats['internal_fragmentation_pct']:.1f}% (Paged blocks)", "Zero Ext. Waste"],
    ]

    headers = ["Metric", "Standard Baseline (HuggingFace)", "Our Custom Engine", "Improvement"]
    
    print("\n" + "=" * 70)
    print("[+] BENCHMARK COMPARISON RESULTS")
    print("=" * 70)
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    print("=" * 70)
    print(f"\nGenerated Output:\n\"{spec_res['output_text']}\"\n")

if __name__ == "__main__":
    main()
