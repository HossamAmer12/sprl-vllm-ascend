from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


# llm = LLM(
#     model="/root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
#     tensor_parallel_size=4,
#     distributed_executor_backend="mp",
#     enforce_eager=True,
#     async_scheduling=False,
# )

llm = LLM(
    model="/root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    async_scheduling=False,
    enforce_eager=True,
    speculative_config={
        "method": "eagle3",
        "model": "/root/.cache/huggingface/hub/models--AngelSlim--Qwen3-1.7B_eagle3/snapshots/94441b48acc5804677ae12259617c83323b543a9",
        "num_speculative_tokens": 2,
    },
)


outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

