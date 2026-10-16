import argparse
import json
import os
import random


def load(path):
    rows = json.load(open(path))
    return {r["id"]: r for r in rows}, [r["id"] for r in rows]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dpo_path", required=True)
    p.add_argument("--sft_path", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--meta_path", default=None, help="eval file with category/expected_behavior, joined by id")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dpo, order = load(args.dpo_path)
    sft, order_sft = load(args.sft_path)
    assert order == order_sft, "the two files must have the same ids in the same order"
    for i in order:
        assert dpo[i]["instruction"] == sft[i]["instruction"], f"prompt mismatch for {i}"
    names = {dpo[i]["generator"] for i in order} | {sft[i]["generator"] for i in order}
    assert len(names) == 2, f"expected exactly two generators, got {names}"

    meta = {}
    if args.meta_path:
        meta = {r["id"]: r for r in json.load(open(args.meta_path))}
        missing = [i for i in order if i not in meta]
        assert not missing, f"{len(missing)} ids missing from meta file, e.g. {missing[:3]}"

    n = len(order)
    dpo_first = [True] * (n // 2) + [False] * (n - n // 2)   # balanced A/B positions
    random.Random(args.seed).shuffle(dpo_first)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    identical = 0
    with open(args.out_path, "w") as f:
        for i, flag in zip(order, dpo_first):
            d, s = dpo[i], sft[i]
            a, b = (d, s) if flag else (s, d)
            identical += a["output"].strip() == b["output"].strip()
            row = {
                "pair_id": f"pair_{i}",
                "id": i,
                "instruction": d["instruction"],
                "response_a": a["output"],
                "response_b": b["output"],
                "hidden": {                      # never shown to the judge
                    "model_a": a["generator"], "model_b": b["generator"],
                    "finished_a": a.get("finished"), "finished_b": b.get("finished"),
                },
            }
            if i in meta:
                row["category"] = meta[i]["category"]
                row["expected_behavior"] = meta[i]["expected_behavior"]
            f.write(json.dumps(row) + "\n")

    print(f"wrote {n} pairs to {args.out_path}")
    print("DPO answer is response A in", sum(dpo_first), "pairs and response B in", n - sum(dpo_first))
    print("pairs where both answers are identical:", identical)


if __name__ == "__main__":
    main()
