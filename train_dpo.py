import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment_core import build_dpo_batch, compute_dpo_loss_from_logps, _token_logps_from_batch
from data_utils import format_prompt, load_jsonl
from train_sft import save_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_path", default="/content/drive/MyDrive/PA2/outputs/sft/best")
    p.add_argument("--train_path", required=True)
    p.add_argument("--val_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_train", type=int, default=15000)
    p.add_argument("--n_val", type=int, default=1000)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--optimizer", choices=["rmsprop", "adamw"], default="rmsprop")
    p.add_argument("--warmup_steps", type=int, default=0, help="0 = fixed lr (manual default)")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--accum", type=int, default=32)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_prompt_length", type=int, default=256)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--val_batch_size", type=int, default=4)
    p.add_argument("--resume_every", type=int, default=50)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--stop_after", type=int, default=0, help="exit early (to test resume)")
    p.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (saves memory)")
    p.add_argument("--save_opt_state", action="store_true", help="also save optimizer state in resume.pt (about 4 GB more)")
    p.add_argument("--resume_dir", default=None, help="folder for resume.pt and metrics.jsonl (use a Drive folder)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def trim(batch):
    """Drop all-padding columns on the right (padding never changes a row's log-prob)."""
    width = int(batch["attention_mask"].sum(dim=1).max().item())
    return {k: v[:, :width] for k, v in batch.items()}


def seq_logps(model, batch):
    token_logps, _ = _token_logps_from_batch(model, trim(batch))
    return token_logps.sum(dim=-1)            # one summed response log-prob per row (float32)


def make_batch(tokenizer, rows, args):
    prompts = [format_prompt(r["instruction"]) for r in rows]
    return build_dpo_batch(
        tokenizer, prompts,
        [r["chosen"] for r in rows], [r["rejected"] for r in rows],
        args.max_length,
    )


@torch.no_grad()
def reference_val_logps(reference, tokenizer, val, args):
    """Reference log-probs on the validation set. Computed once: the reference never changes."""
    out_c, out_r = [], []
    for i in range(0, len(val), args.val_batch_size):
        b = make_batch(tokenizer, val[i:i + args.val_batch_size], args)
        with torch.autocast("cuda", dtype=torch.float16):
            out_c.append(seq_logps(reference, b["chosen"]).float().cpu())
            out_r.append(seq_logps(reference, b["rejected"]).float().cpu())
    return torch.cat(out_c), torch.cat(out_r)


@torch.no_grad()
def evaluate(policy, tokenizer, val, ref_c, ref_r, args):
    policy.eval()
    rw_all, rl_all = [], []
    for i in range(0, len(val), args.val_batch_size):
        chunk = val[i:i + args.val_batch_size]
        b = make_batch(tokenizer, chunk, args)
        with torch.autocast("cuda", dtype=torch.float16):
            pc = seq_logps(policy, b["chosen"]).float().cpu()
            pr = seq_logps(policy, b["rejected"]).float().cpu()
        rw_all.append(args.beta * (pc - ref_c[i:i + len(chunk)]))     # implicit reward, chosen
        rl_all.append(args.beta * (pr - ref_r[i:i + len(chunk)]))     # implicit reward, rejected
    policy.train()
    rw, rl = torch.cat(rw_all), torch.cat(rl_all)
    margin = rw - rl
    return {
        "val_loss": (-F.logsigmoid(margin)).mean().item(),
        "val_reward_acc": (rw > rl).float().mean().item(),             # strict comparison
        "val_reward_margin": margin.mean().item(),
        "val_reward_chosen": rw.mean().item(),
        "val_reward_rejected": rl.mean().item(),
    }


def log_metrics(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def save_resume(path, policy, opt, sched, scaler, step, best, save_opt=False):
    tmp = path + ".tmp"
    state = {"model": policy.state_dict(), "sched": sched.state_dict(),
             "scaler": scaler.state_dict(), "step": step, "best": best}
    if save_opt:
        state["opt"] = opt.state_dict()
    torch.save(state, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    state_dir = args.resume_dir or args.out_dir
    os.makedirs(state_dir, exist_ok=True)
    metrics_path = os.path.join(state_dir, "metrics.jsonl")
    resume_path = os.path.join(state_dir, "resume.pt")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.sft_path)

    def load_and_check(path, n):
        rows = load_jsonl(path)[:n]
        lens = [len(x) for x in tokenizer([format_prompt(r["instruction"]) for r in rows],
                                          add_special_tokens=False).input_ids]
        kept = [r for r, l in zip(rows, lens) if l <= args.max_prompt_length]
        if len(kept) < len(rows):
            print(f"dropped {len(rows) - len(kept)} pairs with prompt > {args.max_prompt_length} tokens")
        return kept

    train = load_and_check(args.train_path, args.n_train)
    val = load_and_check(args.val_path, args.n_val)

    policy = AutoModelForCausalLM.from_pretrained(args.sft_path, dtype=torch.float32).to("cuda")
    policy.config.use_cache = False
    if args.grad_ckpt:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy.train()

    reference = AutoModelForCausalLM.from_pretrained(args.sft_path, dtype=torch.float16).to("cuda")
    reference.config.use_cache = False
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    params = [p for p in policy.parameters() if p.requires_grad]
    if args.optimizer == "rmsprop":
        opt = torch.optim.RMSprop(params, lr=args.lr)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (min(1.0, (s + 1) / args.warmup_steps) if args.warmup_steps > 0 else 1.0)
    )
    scaler = torch.amp.GradScaler("cuda")

    global_batch = args.batch_size * args.accum
    total_steps = math.ceil(len(train) / global_batch)
    order = list(range(len(train)))
    random.Random(args.seed).shuffle(order)

    start_step, best = 0, [-1.0, -1e9]
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        policy.load_state_dict(ckpt["model"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        else:
            print("NOTE: optimizer state was not saved; resuming with a fresh AdamW state")
        sched.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step, best = ckpt["step"], ckpt["best"]
        del ckpt
        print(f"Resuming from step {start_step}")

    print(f"train={len(train)} val={len(val)} | steps={total_steps} | global batch={global_batch} "
          f"| beta={args.beta} lr={args.lr} optimizer={args.optimizer} warmup={args.warmup_steps}")

    ref_c, ref_r = reference_val_logps(reference, tokenizer, val, args)
    if start_step == 0:
        m = evaluate(policy, tokenizer, val, ref_c, ref_r, args)
        print("step 0 | " + " | ".join(f"{k} {v:.4f}" for k, v in m.items()))
        log_metrics(metrics_path, {"step": 0, **m})

    t_start = time.time()
    for step in range(start_step, total_steps):
        group = order[step * global_batch:(step + 1) * global_batch]
        micro = [group[i:i + args.batch_size] for i in range(0, len(group), args.batch_size)]

        opt.zero_grad(set_to_none=True)
        s_loss = s_acc = s_margin = 0.0
        for idxs in micro:
            b = make_batch(tokenizer, [train[i] for i in idxs], args)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                rc = seq_logps(reference, b["chosen"])
                rr = seq_logps(reference, b["rejected"])
            with torch.autocast("cuda", dtype=torch.float16):
                pc = seq_logps(policy, b["chosen"])
                pr = seq_logps(policy, b["rejected"])
            loss = compute_dpo_loss_from_logps(pc, pr, rc.to(pc.device), rr.to(pc.device), args.beta)
            scaler.scale(loss / len(micro)).backward()
            with torch.no_grad():
                margin = args.beta * ((pc - rc.to(pc.device)) - (pr - rr.to(pc.device)))
                s_loss += loss.item() * len(idxs)
                s_acc += (margin > 0).float().sum().item()
                s_margin += margin.sum().item()

        scaler.unscale_(opt)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), args.clip))
        lr_used = opt.param_groups[0]["lr"]
        scaler.step(opt)
        scaler.update()
        sched.step()

        done = step + 1
        n = len(group)
        row = {"step": done, "train_loss": s_loss / n, "train_reward_acc": s_acc / n,
               "train_reward_margin": s_margin / n, "lr": lr_used, "grad_norm": grad_norm}
        log_metrics(metrics_path, row)

        if done % args.log_every == 0 or done == total_steps:
            per_step = (time.time() - t_start) / (done - start_step)
            eta = per_step * (total_steps - done) / 60
            mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"step {done}/{total_steps} | loss {row['train_loss']:.4f} | acc {row['train_reward_acc']:.2f} "
                  f"| margin {row['train_reward_margin']:.3f} | grad_norm {grad_norm:.2f} "
                  f"| {per_step:.1f}s/step | peak mem {mem:.1f}GB | ETA {eta:.0f} min", flush=True)

        if done % args.eval_every == 0 or done == total_steps:
            m = evaluate(policy, tokenizer, val, ref_c, ref_r, args)
            print(f"step {done} | " + " | ".join(f"{k} {v:.4f}" for k, v in m.items()), flush=True)
            log_metrics(metrics_path, {"step": done, **m})
            if (m["val_reward_acc"], m["val_reward_margin"]) > tuple(best):
                best = [m["val_reward_acc"], m["val_reward_margin"]]
                save_model(policy, tokenizer, os.path.join(args.out_dir, "best"))
                print(f"  new best (acc {best[0]:.4f}) -> saved best/", flush=True)

        stopping = args.stop_after and done >= args.stop_after and done < total_steps
        if (args.resume_every and done % args.resume_every == 0 and done < total_steps) or stopping:
            save_resume(resume_path, policy, opt, sched, scaler, done, best, save_opt=args.save_opt_state)
            print(f"  saved resume.pt at step {done}", flush=True)
        if stopping:
            print("stop_after reached, exiting early (this simulates a disconnect)")
            return

    save_model(policy, tokenizer, os.path.join(args.out_dir, "final"))
    print(f"Done. best val reward acc {best[0]:.4f}. Saved final/ in {args.out_dir}")


if __name__ == "__main__":
    main()
