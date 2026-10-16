import torch
import torch.nn.functional as F




def _encode(tokenizer, text):
    """Turn text into a plain list of token ids, with no special tokens added."""
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) > 0 and isinstance(ids[0], (list, tuple)):   # some tokenizers nest the list
        ids = ids[0]
    return list(ids)




def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, labels (prompt + padding masked with -100)."""
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0


    all_ids, all_labels = [], []
    for prompt, response in zip(prompts, responses):
        prompt_ids = _encode(tokenizer, prompt)
        response_ids = _encode(tokenizer, response)
        if eos_id is not None:
            response_ids = response_ids + [eos_id]      # add EOS BEFORE truncating


        ids = (prompt_ids + response_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + response_ids)[:max_length]
        all_ids.append(ids)
        all_labels.append(labels)


    longest = max_length 
    input_ids, attention_mask, labels_out = [], [], []
    for ids, labels in zip(all_ids, all_labels):
        n_pad = longest - len(ids)
        input_ids.append(ids + [pad_id] * n_pad)
        attention_mask.append([1] * len(ids) + [0] * n_pad)
        labels_out.append(labels + [-100] * n_pad)


    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels_out, dtype=torch.long),
    }




def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")




def _token_logps_from_batch(model, batch):
    """Score an already-built batch. Returns (token_logps, mask), zeroed outside the response."""
    device = _model_device(model)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)


    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits if hasattr(out, "logits") else out[0]


    logits = logits[:, :-1, :].float()        # position t predicts token t+1
    targets = labels[:, 1:]                   # so labels shift left by one
    mask = targets != -100                    # True only on response tokens
    safe_targets = targets.masked_fill(~mask, 0)   # -100 is not a valid index


    token_logits = logits.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logits - torch.logsumexp(logits, dim=-1)
    mask = mask.to(token_logps.dtype)
    return token_logps * mask, mask




def _response_token_logps(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    return _token_logps_from_batch(model, batch)




def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row, shape (batch,)."""
    token_logps, _ = _response_token_logps(model, tokenizer, prompts, responses, max_length)
    return token_logps.sum(dim=-1)




def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Scalar loss: mean negative log-prob over all response tokens in the batch."""
    token_logps, mask = _response_token_logps(model, tokenizer, prompts, responses, max_length)
    n_tokens = mask.sum().clamp(min=1.0)      # avoid dividing by zero
    return -token_logps.sum() / n_tokens




def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return {"chosen": batch, "rejected": batch}, each with input_ids/attention_mask/labels."""
    if not (len(prompts) == len(chosen_responses) == len(rejected_responses)):
        raise ValueError(
            "prompts, chosen_responses and rejected_responses must have the same length "
            f"(got {len(prompts)}, {len(chosen_responses)}, {len(rejected_responses)})"
        )
    return {
        "chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length),
        "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length),
    }




def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Scalar mean DPO loss: -log sigmoid(beta * ((pc - pr) - (rc - rr)))."""
    def _prep(x):
        x = torch.as_tensor(x)
        return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


    pc = _prep(policy_chosen_logps)
    pr = _prep(policy_rejected_logps)
    rc = _prep(reference_chosen_logps).detach()     # reference is a constant
    rr = _prep(reference_rejected_logps).detach()


    policy_gap = pc - pr
    reference_gap = rc - rr
    logits = beta * (policy_gap - reference_gap)
    return -F.logsigmoid(logits).mean()




def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Scalar mean DPO loss. Policy keeps gradients; reference runs under no_grad."""
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    chosen, rejected = batch["chosen"], batch["rejected"]


    # Policy: gradients ON
    policy_chosen = _token_logps_from_batch(policy_model, chosen)[0].sum(dim=-1)
    policy_rejected = _token_logps_from_batch(policy_model, rejected)[0].sum(dim=-1)


    # Reference: gradients OFF (same tokens as the policy saw)
    with torch.no_grad():
        reference_chosen = _token_logps_from_batch(reference_model, chosen)[0].sum(dim=-1)
        reference_rejected = _token_logps_from_batch(reference_model, rejected)[0].sum(dim=-1)


    device = policy_chosen.device           # reference may live on another GPU
    return compute_dpo_loss_from_logps(
        policy_chosen,
        policy_rejected,
        reference_chosen.to(device),
        reference_rejected.to(device),
        beta,
    )

