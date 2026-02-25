from vllm import LLM, SamplingParams
import pandas as pd
import re
import io
import logging
from transformers import AutoTokenizer

MODEL_PATH = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

# Set up log capture BEFORE creating the LLM
log_capture = io.StringIO()
handler = logging.StreamHandler(log_capture)
handler.setLevel(logging.INFO)
logging.getLogger("vllm").addHandler(handler)

# Load debug parquet and convert prompts to strings
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
df = pd.read_parquet('/home/hossamamer/cann-recipes-train/agent_rl/hossam_rl/debug.parquet')
# df = pd.read_parquet('/home/hossamamer/cann-recipes-train/agent_rl/hossam_rl/validation.parquet')


print(f"Loaded {len(df)} rows from debug.parquet")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst prompt (raw):\n{df['prompt'].iloc[0]}\n")

# Apply chat template to convert messages -> strings
prompts = []
for _, row in df.iterrows():
    text = tokenizer.apply_chat_template(
        row['prompt'],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompts.append(text)

print(f"First prompt (after chat template):\n{prompts[0][:500]}\n")

prompts = prompts * 10  # 8 * 10 = 80 prompts


sampling_params = SamplingParams(temperature=0.0, top_p=1, top_k=-1, max_tokens=100)

llm = LLM(
    model=MODEL_PATH,
    async_scheduling=False,
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
print("1")
for i, output in enumerate(outputs[:3]):
    print(f"\n[{i+1}] Generated: {output.outputs[0].text!r}")

print("\n" + "="*80)
print("SPECULATIVE DECODING STATS")
print("="*80)

log_output = log_capture.getvalue()
spec_lines = [l for l in log_output.splitlines() if "SpecDecoding metrics" in l]

if spec_lines:
    line = spec_lines[-1]
    print(f"\nRaw log line:\n  {line.strip()}\n")

    def extract_float(pattern, text):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    avg_acceptance_rate = extract_float(r'Avg Draft acceptance rate: ([\d.]+)', line)
    mean_acceptance_len = extract_float(r'Mean acceptance length: ([\d.]+)', line)

    print(f"  Mean acceptance length : {mean_acceptance_len}")
    print(f"  Avg draft acceptance   : {avg_acceptance_rate}%")
else:
    print("\nNo SpecDecoding metrics found.")
    print(log_output[-2000:])