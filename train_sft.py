import argparse
import json
import math
import os
import random
import time


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


from alignment_core import _response_token_logps, compute_sft_loss
from data_utils import format_prompt, load_jsonl




def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--train_path", default="/content/drive/MyDrive/PA2/data/sft_train.jsonl")
    p.add_argument("--val_path", default="/content/drive/MyDrive/PA2/data/sft_val.jsonl")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_train", type=int, default=30000)
    p.add_argument("--n_val", type=int, default=1000)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_frac", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--val_batch_size", type=int, default=4)
    p.add_argument("--resume_every", type=int, default=100)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--stop_after", type=int, default=0, help="exit early (to test resume)")
    p.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (saves memory)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()




def save_model(model, tokenizer, path):
    """Save weights in fp16 (about 1 GB) without touching the fp32 training weights."""
    os.makedirs(path, exist_ok=True)
    half = {k: v.detach().to(torch.float16) for k, v in model.state_dict().items()}
    if model.config.tie_word_embeddings:
        half.pop("lm_head.weight", None)
    model.save_pretrained(path, state_dict=half)
    tokenizer.save_pretrained(path)




def save_resume(path, model, opt, sched, scaler, step, best_val):
    tmp = path + ".tmp"          # write to a temp file first so a disconnect can't corrupt it
    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
         "scaler": scaler.state_dict(), "step": step, "best_val": best_val},
        tmp,
    )
    os.replace(tmp, path)




@torch.no_grad()
def evaluate(model, tokenizer, val, args):
    """Exact token-mean NLL over all response tokens in the validation set."""
    model.eval()
    total_nll, total_tok = 0.0, 0.0
    for i in range(0, len(val), args.val_batch_size):
        chunk = val[i:i + args.val_batch_size]
        prompts = [format_prompt(e["instruction"]) for e in chunk]
        responses = [e["response"] for e in chunk]
        with torch.autocast("cuda", dtype=torch.float16):
            token_logps, mask = _response_token_logps(
                model, tokenizer, prompts, responses, args.max_length
            )
        total_nll += -token_logps.sum().item()
        total_tok += mask.sum().item()
    model.train()
    return total_nll / max(total_tok, 1.0)




def log_metrics(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")




def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, "metrics.jsonl")
    resume_path = os.path.join(args.out_dir, "resume.pt")
    random.seed(args.seed)
    torch.manual_seed(args.seed)


    train = load_jsonl(args.train_path)[: args.n_train]
    val = load_jsonl(args.val_path)[: args.n_val]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float32).to("cuda")
    model.config.use_cache = False
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()


    # Weight decay on matrices only (not biases / norm weights)
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
    )


    global_batch = args.batch_size * args.accum
    total_steps = math.ceil(len(train) / global_batch)
    warmup_steps = max(1, round(args.warmup_frac * total_steps))


    def lr_lambda(step):                       # step = optimizer steps completed so far
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))


    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")


    order = list(range(len(train)))
    random.Random(args.seed).shuffle(order)    # same order every run, so resume is exact


    start_step, best_val = 0, float("inf")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step, best_val = ckpt["step"], ckpt["best_val"]
        del ckpt
        print(f"Resuming from step {start_step}")


    print(f"train={len(train)} val={len(val)} | total optimizer steps={total_steps} "
          f"| warmup={warmup_steps} | global batch={global_batch}")


    if start_step == 0:
        v0 = evaluate(model, tokenizer, val, args)
        print(f"step 0 | val_loss {v0:.4f}")
        log_metrics(metrics_path, {"step": 0, "val_loss": v0})


    t_start = time.time()
    for step in range(start_step, total_steps):
        group = order[step * global_batch:(step + 1) * global_batch]
        micro = [group[i:i + args.batch_size] for i in range(0, len(group), args.batch_size)]


        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for idxs in micro:
            prompts = [format_prompt(train[i]["instruction"]) for i in idxs]
            responses = [train[i]["response"] for i in idxs]
            with torch.autocast("cuda", dtype=torch.float16):
                loss = compute_sft_loss(model, tokenizer, prompts, responses, args.max_length)
            scaler.scale(loss / len(micro)).backward()
            step_loss += loss.item() / len(micro)


        scaler.unscale_(opt)                                   # unscale before clipping
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip))
        lr_used = opt.param_groups[0]["lr"]
        scaler.step(opt)
        scaler.update()
        sched.step()


        done = step + 1
        elapsed = time.time() - t_start
        per_step = elapsed / (done - start_step)
        row = {"step": done, "train_loss": step_loss, "lr": lr_used, "grad_norm": grad_norm}
        log_metrics(metrics_path, row)


        if done % args.log_every == 0 or done == total_steps:
            eta_min = per_step * (total_steps - done) / 60
            mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"step {done}/{total_steps} | loss {step_loss:.4f} | lr {lr_used:.2e} | "
                  f"grad_norm {grad_norm:.2f} | {per_step:.1f}s/step | peak mem {mem:.1f}GB | "
                  f"ETA {eta_min:.0f} min")


        if done % args.eval_every == 0 or done == total_steps:
            v = evaluate(model, tokenizer, val, args)
            print(f"step {done} | val_loss {v:.4f}")
            log_metrics(metrics_path, {"step": done, "val_loss": v})
            if v < best_val:
                best_val = v
                save_model(model, tokenizer, os.path.join(args.out_dir, "best"))
                print(f"  new best val_loss {v:.4f} -> saved best/")


        stopping = args.stop_after and done >= args.stop_after and done < total_steps
        if (args.resume_every and done % args.resume_every == 0 and done < total_steps) or stopping:
            save_resume(resume_path, model, opt, sched, scaler, done, best_val)
            print(f"  saved resume.pt at step {done}")
        if stopping:
            print("stop_after reached, exiting early (this simulates a disconnect)")
            return


    save_model(model, tokenizer, os.path.join(args.out_dir, "final"))
    print(f"Done. best val_loss {best_val:.4f}. Saved final/ in {args.out_dir}")




if __name__ == "__main__":
    main()

