# subset_kl.py
from __future__ import annotations
import torch
from typing import Tuple, Dict

@torch.no_grad()
def pps_sample_indices_batched(
    P: torch.Tensor,   # [N, V], rows sum≈1
    m: int,
    *,
    topk: int = 0,
    min_pi: float = 1e-6,
    gen: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Vectorized PPS: sample m with replacement per row, union with row topk, dedupe.
    Returns:
      idx   : [N, S_max] int64, -1 padded
      pi    : [N, S_max] float32, inclusion probs
      mask  : [N, S_max] bool, valid positions
      info  : diagnostics
    """
    assert m >= 1
    N, V = P.shape
    P = torch.clamp(P, min=0)
    P = P / P.sum(-1, keepdim=True).clamp_min(1e-12)
    P = P.to(torch.float32)

    # Draw m per row in one call
    draws = torch.multinomial(P, num_samples=m, replacement=True, generator=gen)  # [N, m]

    # Row topk
    if topk > 0:
        k = min(topk, V)
        topk_idx = torch.topk(P, k=k, dim=-1).indices                              # [N, k]
        cat_idx = torch.cat([draws, topk_idx], dim=-1)                             # [N, m+k]
    else:
        k = 0
        topk_idx = None
        cat_idx = draws                                                            # [N, m]

    # Sort and dedupe per row
    idx_sorted, _ = torch.sort(cat_idx, dim=-1)                                    # [N, M]
    M = idx_sorted.size(1)
    # keep first, and positions where value changes from previous
    keep = torch.ones_like(idx_sorted, dtype=torch.bool)
    keep[:, 1:] = idx_sorted[:, 1:] != idx_sorted[:, :-1]                          # [N, M]

    # Build padded outputs (S_max ≤ M)
    # Compute π = 1 - (1 - P_i)^m, but π=1 for topk members
    P_sel = P.gather(-1, idx_sorted)                                               # [N, M]
    pi = 1.0 - (1.0 - P_sel).pow(m)                                                # [N, M]
    if k > 0:
        # mark membership in topk per row without torch.isin
        in_topk = (idx_sorted.unsqueeze(-1) == topk_idx.unsqueeze(-2)).any(dim=-1) # [N, M]
        pi = torch.where(in_topk, torch.ones_like(pi), pi)
    pi = pi.clamp_min(min_pi)

    # Compress by masking non-unique positions; keep shape [N, M] with a mask
    idx = idx_sorted.masked_fill(~keep, -1)                                        # [N, M]
    pi = pi.masked_fill(~keep, 0.0)                                                # [N, M]
    mask = keep                                                                     # [N, M]

    info = {
        "avg_S": float(mask.sum(-1).float().mean().item()),
        "max_S": float(mask.sum(-1).max().item()),
    }
    return idx, pi, mask, info

def ht_from_sampled(
    z, P, idx, pi, mask, *, eps=1e-12,
):
    import torch
    N, V = z.shape
    idx_safe = idx.clamp_min(0)

    z_sub = z.gather(-1, idx_safe).to(torch.float32)      # [N,S]
    P_sub = P.gather(-1, idx_safe).to(torch.float32)      # [N,S]

    valid = mask & (pi > 0)
    w = torch.zeros_like(pi, dtype=torch.float32)
    w[valid] = (1.0 / pi[valid])

    msk = mask
    z_sub = z_sub * msk
    P_sub = P_sub * msk
    w = w * msk

    # ---- DEBUG: row stats before math ----
    with torch.no_grad():
        S = msk.sum(-1)                                     # [N]
        rows_empty = (S == 0)
        if rows_empty.any():
            print(f"[ht] empty rows: {rows_empty.nonzero(as_tuple=False).squeeze(-1).tolist()}")
        for name, t in [("z_sub", z_sub), ("P_sub", P_sub), ("w", w)]:
            isn, isi = torch.isnan(t).any().item(), torch.isinf(t).any().item()
            print(f"[ht] {name}: nan={isn} inf={isi} min={t[msk].min().item() if msk.any() else 'NA'} max={t[msk].max().item() if msk.any() else 'NA'}")

    Ehat = (P_sub * z_sub * w).sum(-1)                    # [N]

    # Stable log-partition
    z_masked = z_sub.masked_fill(~msk, float("-inf"))
    c = z_masked.max(dim=-1, keepdim=True).values         # [N,1]
    c = torch.where(torch.isfinite(c), c, torch.zeros_like(c))

    ez_scaled = ((z_sub - c).exp()) * w                   # [N,S]
    Zhat_scaled = ez_scaled.sum(-1)                       # [N]

    with torch.no_grad():
        zero_Z = (Zhat_scaled <= 0)
        if zero_Z.any():
            idxs = zero_Z.nonzero(as_tuple=False).squeeze(-1).tolist()
            print(f"[ht] rows with Zhat_scaled<=0: {idxs}")
            # show a couple rows
            for r in idxs[:3]:
                print(f"[ht] row {r}: S={int(S[r])}, max(z_sub)={float(z_masked[r].max())}, max(w)={float(w[r].max())}")

    logZhat = c.squeeze(-1) + (Zhat_scaled + eps).log()   # [N]

    # Final sanity
    with torch.no_grad():
        for name, t in [("Ehat", Ehat), ("logZhat", logZhat)]:
            print(f"[ht] {name}: nan={torch.isnan(t).any().item()} inf={torch.isinf(t).any().item()} min={float(t.min())} max={float(t.max())}")

    Zhat = torch.zeros_like(Ehat)  # not used
    stats = {"subset/sigmaZ2_over_Z2_proxy": (ez_scaled.pow(2).sum(-1) / (Zhat_scaled.clamp_min(eps)**2)).mean()}
    return Ehat, Zhat, logZhat, stats