from vllm import LLM, SamplingParams
import asyncio
import re
import io
import logging

# Set up log capture BEFORE creating the LLM
log_capture = io.StringIO()
handler = logging.StreamHandler(log_capture)
handler.setLevel(logging.INFO)
vllm_logger = logging.getLogger("vllm")
vllm_logger.addHandler(handler)

# prompts = [
#     "The future of AI is",
#     "Machine learning will",
#     "Artificial intelligence can",
#     "Deep learning is",
#     "Neural networks are",
# ] * 10

prompts = [
    # MT-bench style (multi-turn instruction following)
    # "Write a short essay about the impact of climate change on ocean ecosystems.",
    # "Explain the difference between supervised and unsupervised learning in simple terms.",
    # "Describe the key principles of good software architecture.",
    # "What are the main causes of the French Revolution?",
    # "Summarize the pros and cons of renewable energy sources.",

    # Alpaca style (instruction following)
    # "Give me a step-by-step recipe for making chocolate chip cookies.",
    # "List 5 tips for improving time management skills.",
    # "Explain how to set up a Python virtual environment.",
    # "Write a professional email declining a job offer politely.",
    # "Describe the water cycle in simple terms for a 10-year-old.",

    # HumanEval style (code generation)
    "Write a Python function that checks if a string is a palindrome.",
    "Implement a binary search algorithm in Python.",
    "Write a Python function that flattens a nested list.",
    "Create a Python class for a simple stack data structure.",
    "Write a Python function that counts word frequencies in a text.",

    # GSM8K style (math reasoning)
    # "A store sells apples for $0.50 each. If John buys 12 apples and pays with a $10 bill, how much change does he get?",
    # "A train travels at 60 mph for 2.5 hours. How far does it travel in total?",
    # "If a rectangle has a length of 8cm and a width of 5cm, what is its area and perimeter?",
    # "Sarah has 3 times as many marbles as Tom. If Tom has 15 marbles, how many do they have together?",
    # "A shirt costs $25 and is on sale for 20% off. What is the final price after the discount?",
] * 12  # Repeat 3x to get 60 prompts for better stats

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

llm = LLM(
    model="/root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    async_scheduling=False,
    enforce_eager=True,
    disable_log_stats=False,
    speculative_config={
        "method": "eagle3",
        "model": "/root/.cache/huggingface/hub/models--AngelSlim--Qwen3-1.7B_eagle3/snapshots/94441b48acc5804677ae12259617c83323b543a9",
        "num_speculative_tokens": 2,
    },
)

print("Generating text...")
outputs = llm.generate(prompts, sampling_params)

print("\n" + "="*80)
print("GENERATION RESULTS")
print("="*80)
for i, output in enumerate(outputs[:3]):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"\n[{i+1}] Prompt: {prompt!r}")
    print(f"    Generated: {generated_text!r}")

print("\n" + "="*80)
print("SPECULATIVE DECODING STATS")
print("="*80)

log_output = log_capture.getvalue()
spec_lines = [l for l in log_output.splitlines() if "SpecDecoding metrics" in l]

if spec_lines:
    line = spec_lines[-1]  # Use the final summary line
    print(f"\nRaw log line:\n  {line.strip()}\n")

    def extract_float(pattern, text):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    def extract_int(pattern, text):
        m = re.search(pattern, text)
        return int(m.group(1)) if m else None

    mean_acceptance_len = extract_float(r'Mean acceptance length: ([\d.]+)', line)
    accepted_throughput = extract_float(r'Accepted throughput: ([\d.]+)', line)
    drafted_throughput  = extract_float(r'Drafted throughput: ([\d.]+)', line)
    accepted_tokens     = extract_int(r'Accepted: (\d+)', line)
    drafted_tokens      = extract_int(r'Drafted: (\d+)', line)
    avg_acceptance_rate = extract_float(r'Avg Draft acceptance rate: ([\d.]+)', line)

    pos_match = re.search(r'Per-position acceptance rate: ([\d.,\s]+)', line)
    per_position_rates = []
    if pos_match:
        per_position_rates = [float(x.strip()) for x in pos_match.group(1).split(",") if x.strip()]

    print(f"  Mean acceptance length : {mean_acceptance_len}")
    print(f"  Accepted throughput    : {accepted_throughput} tok/s")
    print(f"  Drafted throughput     : {drafted_throughput} tok/s")
    print(f"  Accepted tokens        : {accepted_tokens}")
    print(f"  Drafted tokens         : {drafted_tokens}")
    print(f"  Avg draft acceptance   : {avg_acceptance_rate}%")
    if per_position_rates:
        print(f"  Per-position rates     :")
        for i, rate in enumerate(per_position_rates):
            print(f"    Position {i}: {rate:.3f} ({rate*100:.1f}%)")

    # Summary recommendation
    if avg_acceptance_rate is not None:
        print()
        if avg_acceptance_rate < 30:
            print(f"  ⚠  Low acceptance rate ({avg_acceptance_rate}%). Consider reducing num_speculative_tokens to 1.")
        elif avg_acceptance_rate > 70:
            print(f"  ✓  High acceptance rate ({avg_acceptance_rate}%). Consider increasing num_speculative_tokens.")
        else:
            print(f"  ✓  Moderate acceptance rate ({avg_acceptance_rate}%). Current config looks reasonable.")

else:
    print("\nNo SpecDecoding metrics found in logs.")
    print("Make sure disable_log_stats=False and check that generation ran long enough to trigger a log interval.")
    print("\n--- Full captured log (for debugging) ---")
    print(log_output[-3000:])  # Print last 3000 chars of logs

print("\n" + "="*80)