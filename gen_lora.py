
import argparse, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--prompts", required=True)
p.add_argument("--out", required=True)
p.add_argument("--adapter", default=None)
p.add_argument("--generator", required=True)
p.add_argument("--limit", type=int, default=None)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--max_new", type=int, default=256)
a = p.parse_args()

NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
tok = AutoTokenizer.from_pretrained(NAME)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).to("cuda")
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter).to("cuda")
model.eval()

rows = json.load(open(a.prompts))
if a.limit:
    rows = rows[:a.limit]
texts = [tok.apply_chat_template([{"role": "user", "content": r["instruction"]}],
                                 tokenize=False, add_generation_prompt=True) for r in rows]
print("example formatted prompt:\n", texts[0])

out = []
for i in range(0, len(rows), a.batch):
    enc = tok(texts[i:i+a.batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False,
                           pad_token_id=tok.pad_token_id)
    for j, r in enumerate(rows[i:i+a.batch]):
        new = g[j][enc["input_ids"].shape[1]:]
        finished = bool((new == tok.eos_token_id).any())
        out.append({"id": r["id"], "instruction": r["instruction"], "output": tok.decode(new, skip_special_tokens=True).strip(),
                    "generator": a.generator, "dataset": "style_eval", "category": r["category"], "finished": finished})
json.dump(out, open(a.out, "w"), indent=1)
print("saved", len(out), "rows to", a.out)
