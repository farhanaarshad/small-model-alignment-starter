import json
import os
from datasets import load_dataset


# One prompt template, used for SFT, DPO and evaluation. Document it in your report.
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"




def format_prompt(instruction):
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())




def load_sft_examples(n_train=30000, n_val=1000, seed=42):
    """Single-turn (instruction, response) pairs from UltraChat. Train and val are disjoint."""
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=seed)


    examples = []
    for row in ds:
        msgs = row["messages"]
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            continue
        instruction = msgs[0]["content"].strip()
        response = msgs[1]["content"].strip()
        if not instruction or not response:
            continue
        examples.append({"instruction": instruction, "response": response})
        if len(examples) == n_train + n_val:
            break
    return examples[:n_train], examples[n_train:]




def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")




def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


