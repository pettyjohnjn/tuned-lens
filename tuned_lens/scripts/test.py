import math, torch
torch.set_default_dtype(torch.float32)
torch.manual_seed(0)

# Config
B, T, V = 3, 4, 500        # small but nontrivial
K = 64
scale_student = 3.0        # make logits moderately sharp
scale_teacher = 4.0
eps = 1e-20

def topk_with_tail_proxy_kl(z: torch.Tensor, logP: torch.Tensor, k: int) -> torch.Tensor:
    assert k > 0 and k <= z.size(-1)
    z = z.float()
    logP = logP.float()
    B, T, V = z.shape
    k = min(k, V)
    n_tail = V - k
    eps = 1e-20

    # Top-K by teacher
    idx = torch.topk(logP, k=k, dim=-1, sorted=False).indices                  # [B,T,K]
    logP_sub = logP.gather(-1, idx)                                            # [B,T,K]
    P_sub = logP_sub.exp()                                                     # [B,T,K]
    z_sub = z.gather(-1, idx)                                                  # [B,T,K]

    # Teacher tail mass
    P_tail = (1.0 - P_sub.sum(dim=-1, keepdim=True)).clamp_min(0.0)           # [B,T,1]

    # Stable tail log-sum-exp via mask (no subtraction)
    sel = torch.zeros((B, T, V), dtype=torch.bool, device=z.device)
    sel.scatter_(-1, idx, True)                                                # mark top-K
    z_tail_masked = z.masked_fill(sel, float("-inf"))                          # tail only
    log_tail_sumexp = torch.logsumexp(z_tail_masked, dim=-1, keepdim=True)     # [B,T,1]

    # Tail proxy logit
    z_tail_proxy = (torch.full_like(log_tail_sumexp, float("-inf"))
                    if n_tail == 0 else (log_tail_sumexp - math.log(n_tail)))

    # Augment student and teacher to K+1 outcomes
    aug_z = torch.cat([z_sub, z_tail_proxy], dim=-1)                           # [B,T,K+1]
    logQ_aug = torch.log_softmax(aug_z, dim=-1)                                # [B,T,K+1]

    logP_tail = P_tail.clamp_min(eps).log()
    P_aug = torch.cat([P_sub, P_tail], dim=-1)
    logP_aug = torch.cat([logP_sub, logP_tail], dim=-1)

    # Mask tail when P_tail == 0
    tail_mask = torch.cat([torch.ones_like(P_sub, dtype=torch.bool), (P_tail > 0)], dim=-1)
    P_aug = torch.where(tail_mask, P_aug, torch.zeros_like(P_aug))
    logP_aug = torch.where(tail_mask, logP_aug, torch.zeros_like(logP_aug))

    kl = (P_aug * (logP_aug - logQ_aug)).sum(dim=-1).mean()

    # Fallback to pure Top-K if non-finite (should not trigger now)
    if not torch.isfinite(kl):
        logP_sub_n = logP_sub - torch.logsumexp(logP_sub, dim=-1, keepdim=True)
        P_sub_n = logP_sub_n.exp()
        logQ_sub_n = torch.log_softmax(z_sub, dim=-1)
        kl = (P_sub_n * (logP_sub_n - logQ_sub_n)).sum(dim=-1).mean()
    return kl

# Toy generator with controllable sharpness and occasional extreme values
def make_batch(B,T,V,scale):
    g = torch.Generator().manual_seed(torch.randint(0, 1_000_000, ()).item())
    x = torch.randn(B,T,V, generator=g) * scale
    # inject a few large logits to stress tails
    mask = torch.rand(B,T,V, generator=g) < 0.001
    x = x + mask * torch.randn(B,T,V, generator=g) * 30.0
    return x

for step in range(200):
    # Teacher logits -> logP
    teacher_logits = make_batch(B,T,V, scale_teacher).requires_grad_(False)
    logP = torch.log_softmax(teacher_logits, dim=-1)

    # Student logits
    student_logits = make_batch(B,T,V, scale_student).requires_grad_(True)

    # Loss
    torch.autograd.set_detect_anomaly(True)
    loss = topk_with_tail_proxy_kl(student_logits, logP, K)

    if not torch.isfinite(loss):
        print(f"[step {step}] loss is NaN")
        break

    loss.backward()
    # Check grads finite and confined to top-k indices
    gnorm = student_logits.grad.nan_to_num().abs().sum().item()
    if not math.isfinite(gnorm):
        print(f"[step {step}] grad non-finite")
        break

    if step % 20 == 0:
        print(f"step={step:03d} loss={loss.item():.6f} grad_L1={gnorm:.3e}")

    # SGD step (toy)
    with torch.no_grad():
        student_logits -= 1e-2 * student_logits.grad
        student_logits.grad.zero_()