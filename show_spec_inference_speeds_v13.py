from vllm import LLM, SamplingParams
import pandas as pd
import re
import io
import logging
from transformers import AutoTokenizer
import time

# MODEL_PATH = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
# DRAFT_MODEL_PATH = "/root/.cache/huggingface/hub/models--AngelSlim--Qwen3-1.7B_eagle3/snapshots/94441b48acc5804677ae12259617c83323b543a9"


# MODEL_PATH = "/tmp/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/"
# DRAFT_MODEL_PATH = "/tmp/.cache/huggingface/models--RedHatAI--Qwen3-8B-speculator.eagle3/snapshots/fd764ef90af0f1f27b7d209ba70a25af54e73bd0"



# MODEL_PATH = "/root/.cache/huggingface/hub/models--Qwen--Qwen2-7B-Instruct/snapshots/f2826a00ceef68f0f2b946d945ecc0477ce4450c"
# DRAFT_MODEL_PATH = "/root/.cache/huggingface/hub/models--yuhuili--EAGLE-Qwen2-7B-Instruct/snapshots/1b9c01aa0c354e16e38ac6dd6b2b0b230be31e9f"


MODEL_PATH = "/tmp/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/"
DRAFT_MODEL_PATH = "/tmp/yuhuili--EAGLE3-LLaMA3.1-Instruct-8B"

# Base model: /tmp/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/
# Eagle model: /tmp/yuhuili--EAGLE3-LLaMA3.1-Instruct-8B

# Base model: /root/.cache/huggingface/hub/models--Qwen--Qwen2-7B-Instruct/snapshots/f2826a00ceef68f0f2b946d945ecc0477ce4450c
# Eagle model: /root/.cache/huggingface/hub/models--yuhuili--EAGLE-Qwen2-7B-Instruct/snapshots/1b9c01aa0c354e16e38ac6dd6b2b0b230be31e9f

# ── Logging ────────────────────────────────────────────────────────────────────
log_capture = io.StringIO()
handler = logging.StreamHandler(log_capture)
handler.setLevel(logging.INFO)
logging.getLogger("vllm").addHandler(handler)

# ── Tokenizer ──────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ── Reconstruct prompts exactly from step 0 log ───────────────────────────────
# These are the exact 8 messages seen at step 0, extracted from the DataProto log
MAX_TOKENS = 4096

raw_messages = [
    [{'role': 'user', 'content': 'Solve the following coding problem using the programming language python:\n\nAn array is called beautiful if all the elements in the array are equal.\n\nYou can transform an array using the following steps any number of times: \n\n  1. Choose two indices i and j (1 ≤ i,j ≤ n), and an integer x (1 ≤ x ≤ a_i). Let i be the source index and j be the sink index. \n  2. Decrease the i-th element by x, and increase the j-th element by x. The resulting values at i-th and j-th index are a_i-x and a_j+x respectively. \n  3. The cost of this operation is x ⋅ |j-i| . \n  4. Now the i-th index can no longer be the sink and the j-th index can no longer be the source. \n\nThe total cost of a transformation is the sum of all the costs in step 3.\n\nFor example, array [0, 2, 3, 3] can be transformed into a beautiful array [2, 2, 2, 2] with total cost 1 ⋅ |1-3| + 1 ⋅ |1-4| = 5.\n\nAn array is called balanced, if it can be transformed into a beautiful array, and the cost of such transformation is uniquely defined. In other words, the minimum cost of transformation into a beautiful array equals the maximum cost.\n\nYou are given an array a_1, a_2, …, a_n of length n, consisting of non-negative integers. Your task is to find the number of balanced arrays which are permutations of the given array. Two arrays are considered different, if elements at some position differ. Since the answer can be large, output it modulo 10^9 + 7.\n\nInput\n\nThe first line contains a single integer n (1 ≤ n ≤ 10^5) — the size of the array. \n\nThe second line contains n integers a_1, a_2, …, a_n (0 ≤ a_i ≤ 10^9).\n\nOutput\n\nOutput a single integer — the number of balanced permutations modulo 10^9+7.\n\nExamples\n\nInput\n\n\n3\n1 2 3\n\n\nOutput\n\n\n6\n\nInput\n\n\n4\n0 4 0 4\n\n\nOutput\n\n\n2\n\nInput\n\n\n5\n0 11 12 13 14\n\n\nOutput\n\n\n120\n\nNote\n\nIn the first example, [1, 2, 3] is a valid permutation as we can consider the index with value 3 as the source and index with value 1 as the sink. Thus, after conversion we get a beautiful array [2, 2, 2], and the total cost would be 2. We can show that this is the only transformation of this array that leads to a beautiful array. Similarly, we can check for other permutations too.\n\nIn the second example, [0, 0, 4, 4] and [4, 4, 0, 0] are balanced permutations.\n\nIn the third example, all permutations are balanced.\n\nThe input will be given via stdin and the output should be printed to stdout by your code.\n\nNow solve the problem by providing the code.'}],
    [{'role': 'user', 'content': 'Solve the following coding problem using the programming language python:\n\nWe have A balls with the string S written on each of them and B balls with the string T written on each of them.\nFrom these balls, Takahashi chooses one with the string U written on it and throws it away.\nFind the number of balls with the string S and balls with the string T that we have now.\n\nConstraints\n\n* S, T, and U are strings consisting of lowercase English letters.\n* The lengths of S and T are each between 1 and 10 (inclusive).\n* S \\not= T\n* S=U or T=U.\n* 1 \\leq A,B \\leq 10\n* A and B are integers.\n\nInput\n\nInput is given from Standard Input in the following format:\n\n\nS T\nA B\nU\n\n\nOutput\n\nPrint the answer, with space in between.\n\nExamples\n\nInput\n\nred blue\n3 4\nred\n\n\nOutput\n\n2 4\n\n\nInput\n\nred blue\n5 5\nblue\n\n\nOutput\n\n5 4\n\nThe input will be given via stdin and the output should be printed to stdout by your code.\n\nNow solve the problem by providing the code.'}],
    [{'role': 'user', 'content': "Solve the following coding problem using the programming language python:\n\nYear 2118. Androids are in mass production for decades now, and they do all the work for humans. But androids have to go to school to be able to solve creative tasks. Just like humans before.\n\nIt turns out that high school struggles are not gone. If someone is not like others, he is bullied. Vasya-8800 is an economy-class android which is produced by a little-known company. His design is not perfect, his characteristics also could be better. So he is bullied by other androids.\n\nOne of the popular pranks on Vasya is to force him to compare $x^y$ with $y^x$. Other androids can do it in milliseconds while Vasya's memory is too small to store such big numbers.\n\nPlease help Vasya! Write a fast program to compare $x^y$ with $y^x$ for Vasya, maybe then other androids will respect him.\n\n\n-----Input-----\n\nOn the only line of input there are two integers $x$ and $y$ ($1 \\le x, y \\le 10^{9}$).\n\n\n-----Output-----\n\nIf $x^y < y^x$, then print '<' (without quotes). If $x^y > y^x$, then print '>' (without quotes). If $x^y = y^x$, then print '=' (without quotes).\n\n\n-----Examples-----\nInput\n5 8\n\nOutput\n>\n\nInput\n10 3\n\nOutput\n<\n\nInput\n6 6\n\nOutput\n=\n\n\n\n-----Note-----\n\nIn the first example $5^8 = 5 \\cdot 5 \\cdot 5 \\cdot 5 \\cdot 5 \\cdot 5 \\cdot 5 \\cdot 5 = 390625$, and $8^5 = 8 \\cdot 8 \\cdot 8 \\cdot 8 \\cdot 8 = 32768$. So you should print '>'.\n\nIn the second example $10^3 = 1000 < 3^{10} = 59049$.\n\nIn the third example $6^6 = 46656 = 6^6$.\n\nThe input will be stdin and you should print your solution to stdout\n\n\nNow solve the problem and return the code."}],
    [{'role': 'user', 'content': "Solve the following coding problem using the programming language python:\n\nMike decided to teach programming to children in an elementary school. He knows that it is not an easy task to interest children in that age to code. That is why he decided to give each child two sweets.\n\nMike has $n$ sweets with sizes $a_1, a_2, \\ldots, a_n$. All his sweets have different sizes. That is, there is no such pair $(i, j)$ ($1 \\leq i, j \\leq n$) such that $i \\ne j$ and $a_i = a_j$.\n\nSince Mike has taught for many years, he knows that if he gives two sweets with sizes $a_i$ and $a_j$ to one child and $a_k$ and $a_p$ to another, where $(a_i + a_j) \\neq (a_k + a_p)$, then a child who has a smaller sum of sizes will be upset. That is, if there are two children who have different sums of sweets, then one of them will be upset. Apparently, Mike does not want somebody to be upset. \n\nMike wants to invite children giving each of them two sweets. Obviously, he can't give one sweet to two or more children. His goal is to invite as many children as he can. \n\nSince Mike is busy preparing to his first lecture in the elementary school, he is asking you to find the maximum number of children he can invite giving each of them two sweets in such way that nobody will be upset.\n\n\n-----Input-----\n\nThe first line contains one integer $n$ ($2 \\leq n \\leq 1\\,000$) — the number of sweets Mike has.\n\nThe second line contains $n$ integers $a_1, a_2, \\ldots, a_n$ ($1 \\leq a_i \\leq 10^5$) — the sizes of the sweets. It is guaranteed that all integers are distinct.\n\n\n-----Output-----\n\nPrint one integer — the maximum number of children Mike can invite giving each of them two sweets in such way that nobody will be upset.\n\n\n-----Examples-----\nInput\n8\n1 8 3 11 4 9 2 7\n\nOutput\n3\n\nInput\n7\n3 1 7 11 9 2 12\n\nOutput\n2\n\n\n\n-----Note-----\n\nIn the first example, Mike can give $9+2=11$ to one child, $8+3=11$ to another one, and $7+4=11$ to the third child. Therefore, Mike can invite three children. Note that it is not the only solution.\n\nIn the second example, Mike can give $3+9=12$ to one child and $1+11$ to another one. Therefore, Mike can invite two children. Note that it is not the only solution.\n\nThe input will be stdin and you should print your solution to stdout\n\n\nNow solve the problem and return the code."}],
    [{'role': 'user', 'content': "Solve the following coding problem using the programming language python:\n\nCodeforces user' handle color depends on his rating — it is red if his rating is greater or equal to 2400; it is orange if his rating is less than 2400 but greater or equal to 2200, etc. Each time participant takes part in a rated contest, his rating is changed depending on his performance.\n\nAnton wants the color of his handle to become red. He considers his performance in the rated contest to be good if he outscored some participant, whose handle was colored red before the contest and his rating has increased after it.\n\nAnton has written a program that analyses contest results and determines whether he performed good or not. Are you able to do the same?\n\nInput\n\nThe first line of the input contains a single integer n (1 ≤ n ≤ 100) — the number of participants Anton has outscored in this contest .\n\nThe next n lines describe participants results: the i-th of them consists of a participant handle namei and two integers beforei and afteri ( - 4000 ≤ beforei, afteri ≤ 4000) — participant's rating before and after the contest, respectively. Each handle is a non-empty string, consisting of no more than 10 characters, which might be lowercase and uppercase English letters, digits, characters «_» and «-» characters.\n\nIt is guaranteed that all handles are distinct.\n\nOutput\n\nPrint «YES» (quotes for clarity), if Anton has performed good in the contest and «NO» (quotes for clarity) otherwise.\n\nExamples\n\nInput\n\n3\nBurunduk1 2526 2537\nBudAlNik 2084 2214\nsubscriber 2833 2749\n\n\nOutput\n\nYES\n\nInput\n\n3\nApplejack 2400 2400\nFluttershy 2390 2431\nPinkie_Pie -2500 -2450\n\n\nOutput\n\nNO\n\nNote\n\nIn the first sample, Anton has outscored user with handle Burunduk1, whose handle was colored red before the contest and his rating has increased after the contest.\n\nIn the second sample, Applejack's rating has not increased after the contest, while both Fluttershy's and Pinkie_Pie's handles were not colored red before the contest.\n\nThe input will be given via stdin and the output should be printed to stdout by your code.\n\nNow solve the problem by providing the code."}],
    [{'role': 'user', 'content': 'Solve the following coding problem using the programming language python:\n\nThere are N children, numbered 1, 2, \\ldots, N.\n\nThey have decided to share K candies among themselves. Here, for each i (1 \\leq i \\leq N), Child i must receive between 0 and a_i candies (inclusive). Also, no candies should be left over.\n\nFind the number of ways for them to share candies, modulo 10^9 + 7. Here, two ways are said to be different when there exists a child who receives a different number of candies.\n\nConstraints\n\n* All values in input are integers.\n* 1 \\leq N \\leq 100\n* 0 \\leq K \\leq 10^5\n* 0 \\leq a_i \\leq K\n\nInput\n\nInput is given from Standard Input in the following format:\n\n\nN K\na_1 a_2 \\ldots a_N\n\n\nOutput\n\nPrint the number of ways for the children to share candies, modulo 10^9 + 7.\n\nExamples\n\nInput\n\n3 4\n1 2 3\n\n\nOutput\n\n5\n\n\nInput\n\n1 10\n9\n\n\nOutput\n\n0\n\n\nInput\n\n2 0\n0 0\n\n\nOutput\n\n1\n\n\nInput\n\n4 100000\n100000 100000 100000 100000\n\n\nOutput\n\n665683269\n\nThe input will be given via stdin and the output should be printed to stdout by your code.'}],
    [{'role': 'user', 'content': 'Solve the following coding problem using the programming language python:\n\nOn the Literature lesson Sergei noticed an awful injustice, it seems that some students are asked more often than others.\n\nSeating in the class looks like a rectangle, where n rows with m pupils in each. \n\nThe teacher asks pupils in the following order: at first, she asks all pupils from the first row in the order of their seating, then she continues to ask pupils from the next row. If the teacher asked the last row, then the direction of the poll changes, it means that she asks the previous row. The order of asking the rows looks as follows: the 1-st row, the 2-nd row, ..., the n - 1-st row, the n-th row, the n - 1-st row, ..., the 2-nd row, the 1-st row, the 2-nd row, ...\n\nThe order of asking of pupils on the same row is always the same: the 1-st pupil, the 2-nd pupil, ..., the m-th pupil.\n\nDuring the lesson the teacher managed to ask exactly k questions from pupils in order described above. Sergei seats on the x-th row, on the y-th place in the row. Sergei decided to prove to the teacher that pupils are asked irregularly, help him count three values:  the maximum number of questions a particular pupil is asked,  the minimum number of questions a particular pupil is asked,  how many times the teacher asked Sergei. \n\nIf there is only one row in the class, then the teacher always asks children from this row.\n\n\n-----Input-----\n\nThe first and the only line contains five integers n, m, k, x and y (1 ≤ n, m ≤ 100, 1 ≤ k ≤ 10^18, 1 ≤ x ≤ n, 1 ≤ y ≤ m).\n\n\n-----Output-----\n\nPrint three integers:  the maximum number of questions a particular pupil is asked,  the minimum number of questions a particular pupil is asked,  how many times the teacher asked Sergei. \n\n\n-----Examples-----\nInput\n1 3 8 1 1\n\nOutput\n3 2 3\nInput\n4 2 9 4 2\n\nOutput\n2 1 1\nInput\n5 5 25 4 3\n\nOutput\n1 1 1\nInput\n100 100 1000000000000000000 100 100\n\nOutput\n101010101010101 50505050505051 50505050505051\n\n\n-----Note-----\n\nThe order of asking pupils in the first test is as described above.\n\nThe input will be stdin and you should print your solution to stdout\n\n\nNow solve the problem and return the code.'}],
    [{'role': 'user', 'content': 'Solve the following coding problem using the programming language python:\n\nYou are given a graph with $3 \\cdot n$ vertices and $m$ edges. You are to find a matching of $n$ edges, or an independent set of $n$ vertices.\n\nA set of edges is called a matching if no two edges share an endpoint.\n\nA set of vertices is called an independent set if no two vertices are connected with an edge.\n\n\n-----Input-----\n\nThe first line contains a single integer $T \\ge 1$ — the number of graphs you need to process. The description of $T$ graphs follows.\n\nThe first line of description of a single graph contains two integers $n$ and $m$, where $3 \\cdot n$ is the number of vertices, and $m$ is the number of edges in the graph ($1 \\leq n \\leq 10^{5}$, $0 \\leq m \\leq 5 \\cdot 10^{5}$).\n\nEach of the next $m$ lines contains two integers $v_i$ and $u_i$ ($1 \\leq v_i, u_i \\leq 3 \\cdot n$), meaning that there is an edge between vertices $v_i$ and $u_i$.\n\nIt is guaranteed that there are no self-loops and no multiple edges in the graph.\n\nIt is guaranteed that the sum of all $n$ over all graphs in a single test does not exceed $10^{5}$, and the sum of all $m$ over all graphs in a single test does not exceed $5 \\cdot 10^{5}$.\n\n\n-----Output-----\n\nPrint your answer for each of the $T$ graphs. Output your answer for a single graph in the following format.\n\nIf you found a matching of size $n$, on the first line print "Matching" (without quotes), and on the second line print $n$ integers — the indices of the edges in the matching. The edges are numbered from $1$ to $m$ in the input order.\n\nIf you found an independent set of size $n$, on the first line print "IndSet" (without quotes), and on the second line print $n$ integers — the indices of the vertices in the independent set.\n\nIf there is no matching and no independent set of the specified size, print "Impossible" (without quotes).\n\nYou can print edges and vertices in any order.\n\nIf there are several solutions, print any. In particular, if there are both a matching of size $n$, and an independent set of size $n$, then you should print exactly one of such matchings or exactly one of such independent sets.\n\n\n-----Example-----\nInput\n4\n1 2\n1 3\n1 2\n...\n\nOutput\nMatching\n2\nIndSet\n1\n...\n\nThe input will be stdin and you should print your solution to stdout\n\n\nNow solve the problem and return the code.'}],
]

prompts = []
for messages in raw_messages:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompts.append(text)

print(f"\nNum prompts: {len(prompts)}")
for i, p in enumerate(prompts):
    toks = tokenizer(p, return_tensors='pt')['input_ids'].shape[1]
    print(f"  [{i}] prompt tokens: {toks}")

# ── Match veRL sampling params exactly ────────────────────────────────────────
# From sh: temperature=0.9, no explicit top_p/top_k set in veRL
# veRL warning said: min_p, logit_bias, min_tokens won't work with spec decode
# So keep it clean — just temperature, no top_k/top_p overrides
sampling_params = SamplingParams(
    temperature=0.9,
    top_p=1.0,
    top_k=-1,
    max_tokens=MAX_TOKENS,
)

print(f"\nSamplingParams: {sampling_params}")

# ── Init LLM matching veRL engine kwargs ──────────────────────────────────────
# From veRL sh: tensor_model_parallel_size=1, gpu_memory_utilization=0.7,
# max_num_batched_tokens=6144, max_model_len=6144
# NOT setting enforce_eager (veRL doesn't set it)
# NOT setting async_scheduling=False (veRL uses async mode)
# llm = LLM(
#     model=MODEL_PATH,
#     tensor_parallel_size=1,
#     gpu_memory_utilization=0.7,
#     max_num_batched_tokens=6144,
#     max_model_len=6144,
#     disable_log_stats=False,
#     speculative_config={
#         "method": "eagle3",
#         "model": DRAFT_MODEL_PATH,
#         "num_speculative_tokens": 2,
#     },
#     # async_scheduling intentionally NOT set — let it default like veRL does
# )

llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.7,
    max_num_batched_tokens=6144,
    max_model_len=6144,
    disable_log_stats=False,
    speculative_config={
        "model": DRAFT_MODEL_PATH,
        "num_speculative_tokens": 3,
    },
    # async_scheduling intentionally NOT set — let it default like veRL does
)


# llm = LLM(
#     model=MODEL_PATH,
#     tensor_parallel_size=1,
#     gpu_memory_utilization=0.7,
#     max_num_batched_tokens=6144,
#     max_model_len=6144,
#     disable_log_stats=False,
# )

print("\nWarm up...")
# Warm-up (optional single pass to reduce cold-start bias)
_ = llm.generate(prompts[:1], SamplingParams(temperature=0.9, max_tokens=16))

print("\nGenerating...")

t_start = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)
t_end = time.perf_counter()

elapsed = t_end - t_start
total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
throughput = total_tokens / elapsed


print("\n" + "="*80)
print("GENERATION RESULTS")
print("="*80)
for i, output in enumerate(outputs):
    text = output.outputs[0].text
    print(f"\n[{i}] tokens={len(output.outputs[0].token_ids)} text={text[:200]!r}")

print("\n" + "="*80)
print("SPECULATIVE DECODING STATS")
print("="*80)

log_output = log_capture.getvalue()
spec_lines = [l for l in log_output.splitlines() if "SpecDecoding metrics" in l]

if spec_lines:
    for line in spec_lines:
        print(f"\n{line.strip()}")

    line = spec_lines[-1]
    def extract_float(pattern, text):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    acc = extract_float(r'Avg Draft acceptance rate: ([\d.]+)', line)
    mal = extract_float(r'Mean acceptance length: ([\d.]+)', line)
    print(f"\n  Avg draft acceptance rate : {acc}%")
    print(f"  Mean acceptance length    : {mal}")
else:
    print("\nNo SpecDecoding metrics found in logs.")
    # Print last bit of logs for debugging
    all_lines = log_output.splitlines()
    print("\nLast 30 log lines:")
    for l in all_lines[-30:]:
        print(f"  {l}")

print(f"  ✓ Done in {elapsed:.2f}s | {total_tokens} tokens | {throughput:.1f} tok/s")