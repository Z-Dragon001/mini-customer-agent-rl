"""Memory-bounded vocabulary projection with unchanged logprob/entropy math."""


def probability_diagnostics(inputs, new_logprobs, epsilon=.2):
    """Measure inference/trainer agreement without changing the loss or graph."""
    import torch
    with torch.no_grad():
        old = inputs['logprobs'][:, 1:].float()
        new = new_logprobs[:, :-1].detach().float()
        mask = inputs['assistant_mask'][:, 1:].bool() & torch.isfinite(old)
        if not mask.any():
            return {}
        delta = new[mask] - old[mask]
        ratio = torch.exp(delta.clamp(-20, 20))
        metrics = {'old_policy_logprob_mae': delta.abs().mean().item(),
                'old_policy_logprob_bias': delta.mean().item(),
                'old_policy_ratio_outside_clip': ((ratio < 1-epsilon) | (ratio > 1+epsilon)).float().mean().item()}
        if 'input_pos' in inputs:
            packed = inputs['input_pos'][:, :-1]
            sequential = torch.arange(packed.shape[1], device=packed.device)[None, :]
            metrics['packed_position_reset_fraction'] = (packed != sequential)[mask].float().mean().item()
        return metrics


def checkpointed_logprobs(lm_head_t, hidden_states, next_input_ids, chunk_size, *, fp32_output=False):
    import torch
    import torch.utils.checkpoint as checkpoint_module
    # Unsloth 2025.12's wrapper omits PyTorch 2.9's new `early_stop`
    # positional argument in the non-reentrant implementation. Its saved
    # original retains the correct signature; model-layer checkpointing stays
    # under Unsloth's control.
    checkpoint = getattr(checkpoint_module, "_old_checkpoint", checkpoint_module.checkpoint)

    lm_head_t = lm_head_t.to(hidden_states.dtype)

    def project(states, head, targets):
        logits = torch.matmul(states, head)
        if fp32_output:
            logits = logits.float()
        selected = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        normalizer = torch.logsumexp(logits, dim=-1)
        full_logprobs = logits - normalizer.unsqueeze(-1)
        entropy = (-torch.exp(full_logprobs) * full_logprobs).sum(dim=-1)
        return selected - normalizer, entropy

    logprobs, entropies = [], []
    for start in range(0, hidden_states.shape[1], chunk_size):
        states = hidden_states[:, start:start + chunk_size, :]
        targets = next_input_ids[:, start:start + chunk_size]
        if torch.is_grad_enabled() and (states.requires_grad or lm_head_t.requires_grad):
            values = checkpoint(project, states, lm_head_t, targets,
                                use_reentrant=False, preserve_rng_state=False)
        else:
            values = project(states, lm_head_t, targets)
        logprobs.append(values[0])
        entropies.append(values[1])
    return torch.cat(logprobs, dim=1), torch.cat(entropies, dim=1)
