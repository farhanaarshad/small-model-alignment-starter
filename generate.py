import argparse
import json
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import format_prompt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--prompts_path", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--generator", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--no_repeat_ngram_size", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="only the first N prompts (for tests)")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(42)
    torch.manual_seed(42)

    rows = json.load(open(args.prompts_path))
    if args.limit:
        rows = rows[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model_path)
    tok.padding_side = "left"                      # decoder-only models need left padding to generate
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float16).to("cuda")
    model.eval()

    gen_kwargs = dict(
        do_sample=False,
        temperature=None, top_p=None, top_k=None,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    if args.no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    results = []
    t0 = time.time()
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start:start + args.batch_size]
        instructions = [r.get("instruction") or r.get("prompt") for r in chunk]
        prompts = [format_prompt(x) for x in instructions]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        new_tokens = out[:, enc.input_ids.shape[1]:]
        texts = tok.batch_decode(new_tokens, skip_special_tokens=True)
        finished = (new_tokens == tok.eos_token_id).any(dim=1).tolist()
        for r, ins, text, fin in zip(chunk, instructions, texts, finished):
            results.append({
                "id": r.get("id", f"{args.dataset}_{len(results)}"),
                "instruction": ins,
                "output": text.strip(),
                "generator": args.generator,
                "dataset": args.dataset,
                "finished": fin,
            })
        done = min(start + args.batch_size, len(rows))
        print(f"{done}/{len(rows)} done | {time.time() - t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(results, f, indent=1)

    share = sum(r["finished"] for r in results) / len(results)
    print(f"saved {len(results)} outputs to {args.out_path}")
    print(f"share that ended with EOS before the length limit: {share:.2f}")
    for r in results[:3]:
        print("\n--- PROMPT:", r["instruction"][:120])
        print("--- OUTPUT:", r["output"][:400])


if __name__ == "__main__":
    main()
