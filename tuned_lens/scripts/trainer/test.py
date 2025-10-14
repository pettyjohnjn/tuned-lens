import torch
from subset_kl import pps_sample_indices, ht_estimators_from_subset

torch.manual_seed(7)
N, V = 3, 200
P = torch.softmax(torch.randn(N, V), -1).to(torch.float32)
z = torch.randn(N, V) * 0.5  # student logits

m, topk = 30, 5
idx, pi, mask, _ = pps_sample_indices(P, m, topk=topk)

Ehat, Zhat, logZhat, stats = ht_estimators_from_subset(z, P, idx, pi, mask)

# References
E_ref = (P * z).sum(-1)
Z_ref = z.exp().sum(-1)
logZ_ref = Z_ref.log()

print("Ehat vs E_ref (mean abs diff):", (Ehat - E_ref).abs().mean().item())
print("Zhat mean:", Zhat.mean().item(), "Z_ref mean:", Z_ref.mean().item())
print("E[logZhat] ≤ logZ? sample means:", logZhat.mean().item(), "<=", logZ_ref.mean().item())
print("stats:", {k: float(v) for k, v in stats.items()})

# Expect small Ehat error. logZhat is biased low; average should be ≤ reference on expectation.

# Single row gradient spot-check
z = torch.randn(1, V, requires_grad=True)
P = torch.softmax(torch.randn(1, V), -1)
idx, pi, mask, _ = pps_sample_indices(P, m=20, topk=5)
Ehat, Zhat, logZhat, _ = ht_estimators_from_subset(z, P, idx, pi, mask)

loss = -(Ehat - logZhat).mean()
loss.backward()

# Grad nonzero only on sampled indices
sampled = idx[0, mask[0]]
nz_idx = torch.nonzero(z.grad[0]).squeeze(-1)
assert set(nz_idx.tolist()).issubset(set(sampled.tolist()))
print("grad check passed; non-sampled grads zero:", nz_idx.numel(), "≤", sampled.numel())