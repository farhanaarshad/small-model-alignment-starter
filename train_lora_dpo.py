
import argparse, json, os, random
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    p.add_argument("--data_path", default="data/persona_lora/ramsay_style_preferences.json")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_val", type=int, default=20)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def to_conv(r):
    return {
        "prompt": [{"role": "user", "content": r["prompt"]}],
        "chosen": [{"role": "assistant", "content": r["chosen"]}],
        "rejected": [{"role": "assistant", "content": r["rejected"]}],
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = json.load(open(args.data_path))
    random.Random(args.seed).shuffle(rows)
    val_rows, train_rows = rows[:args.n_val], rows[args.n_val:]
    json.dump(val_rows, open(f"{args.out_dir}/heldout_val_rows.json", "w"), indent=1)
    train_ds = Dataset.from_list([to_conv(r) for r in train_rows])
    val_ds = Dataset.from_list([to_conv(r) for r in val_rows])
    print(f"train rows: {len(train_rows)} | held-out val rows: {len(val_rows)}")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float32)

    lora = LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    cfg = DPOConfig(
        output_dir=args.out_dir, beta=args.beta, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum, num_train_epochs=args.epochs, max_steps=args.max_steps,
        max_length=args.max_length, lr_scheduler_type="cosine", warmup_ratio=0.1,
        logging_steps=1, eval_strategy="no", save_strategy="no", report_to="none",
        fp16=True, bf16=False, seed=args.seed,
    )
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=val_ds,
                         processing_class=tok, peft_config=lora)
    try:
        trainer.model.print_trainable_parameters()
    except Exception as e:
        print("could not print trainable params:", e)

    show = lambda m: {k: round(v, 4) for k, v in m.items() if isinstance(v, (int, float))}
    pre = trainer.evaluate()
    print("EVAL BEFORE TRAINING:", show(pre))
    trainer.train()
    post = trainer.evaluate()
    print("EVAL AFTER TRAINING:", show(post))

    trainer.model.save_pretrained(f"{args.out_dir}/adapter")
    tok.save_pretrained(f"{args.out_dir}/adapter")
    json.dump({"args": vars(args), "eval_before": show(pre), "eval_after": show(post),
               "log_history": trainer.state.log_history},
              open(f"{args.out_dir}/lora_run_summary.json", "w"), indent=1)
    print("saved adapter and summary to", args.out_dir)


if __name__ == "__main__":
    main()
