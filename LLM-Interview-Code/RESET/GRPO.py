import torch
def compute_grpo_advantages(rewards):
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    advantages = (rewards -mean) / (std + 1e-8)
    return advantages

def grpo_loss(old_log_probs, new_log_probs, advantages, clip_epsilon=0.2, beta=0.01, ref_kl=None):
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages
    policy_loss = -torch.mean(surrogate1, surrogate2)
    if ref_kl is not None:
        return (policy_loss + beta * ref_kl).mean()
    return policy_loss.mean()

def compute_grpo_penalty(log_probs, ref_log_probs):
    ratio = torch.exp(ref_log_probs - log_probs)
    kl = ratio - 1 - (ref_log_probs - log_probs)
    return kl.mean()