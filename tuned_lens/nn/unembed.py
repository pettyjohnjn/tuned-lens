"""Provides a class for mapping transformer hidden states to logits (and vice versa)."""
import copy
from dataclasses import dataclass
from typing import Literal, Optional, Union, cast

try:
    # Needed for the docs to build without complaining
    import transformer_lens as tl  # noqa: F401

    _transformer_lens_available = True
except ImportError:
    _transformer_lens_available = False

import torch
from torch.distributions import Distribution

from tuned_lens import model_surgery
from tuned_lens.utils import tensor_hash


@dataclass
class InversionOutput:
    """Output of `Unemebd.invert`."""

    preimage: torch.Tensor
    grad_norm: torch.Tensor
    kl: torch.Tensor
    loss: torch.Tensor
    nfev: int


class Unembed(torch.nn.Module):
    """Module that maps transformer hidden states to logits (and vice versa)."""

    final_norm: model_surgery.Norm
    unembedding: torch.nn.Linear

    def __init__(
        self,
        model: model_surgery.Model,
    ):
        """Initialize unmebed.

        Args:
            model: A HuggingFace model from which to extract the unembedding matrix.
        """
        super().__init__()
        # final_norm = model_surgery.get_final_norm(model)
        # unembedding_matrix = model_surgery.get_unembedding_matrix(model)

        # self.final_norm = copy.deepcopy(final_norm)
        # self.unembedding = copy.deepcopy(unembedding_matrix)

        self.final_norm = copy.deepcopy(model_surgery.get_final_norm(model))
        self.unembedding = model_surgery.get_unembedding_matrix(model)
        for p in self.unembedding.parameters():
            p.requires_grad_(False)

        # In general we don't want to finetune the unembed operation.
        self.requires_grad_(False)

    def unembedding_hash(self) -> str:
        """Hash the unmbedding matrix to identify the model."""
        parameter = self.unembedding.weight.data.detach().cpu().float().numpy()
        return tensor_hash(parameter)

    # def forward(self, h: torch.Tensor) -> torch.Tensor:
    #     """Convert hidden states into logits."""
    #     return self.unembedding(self.final_norm(h))

    def forward(
        self,
        h: torch.Tensor,                         # (B,T,d)
        *,                                       # force keyword
        idx_subset: Optional[torch.Tensor] = None   # (B,T,k) or None
    ) -> torch.Tensor:
        """
        Convert hidden states to logits.
        If `idx_subset` is given, compute only those vocab rows and
        return a (B,T,k) tensor; otherwise return (B,T,V).
        """
        h = self.final_norm(h)                   # (B,T,d)

        if idx_subset is None:
            # Full-vocab path – unchanged
            return self.unembedding(h)

        # ---- subset path ----
        B, T, k = idx_subset.shape
        d       = h.size(-1)

        h_flat   = h.reshape(-1, d)              # (B*T,d)
        idx_flat = idx_subset.reshape(-1, k)     # (B*T,k)

        uniq_idx, inverse = torch.unique(idx_flat, return_inverse=True)
        W_sel   = self.unembedding.weight[uniq_idx]   # (u,d)
        logits_sel = h_flat @ W_sel.T                 # (B*T,u)

        logits_k = torch.gather(
            logits_sel, 1, inverse.view(-1, k)
        ).view(B, T, k)
        return logits_k

    def invert(
        self,
        logits: torch.Tensor,
        *,
        h0: Optional[torch.Tensor] = None,
        max_iter: int = 1000,
        optimizer: Literal["lbfgs", "sgd"] = "lbfgs",
        prior_weight: float = 0.0,
        prior: Optional[Distribution] = None,
        step_size: float = 1.0,
        tol: float = 1e-3,
        weight: Optional[torch.Tensor] = None,
    ) -> InversionOutput:
        """Project logits onto the image of the unemebed operation.

        When the hidden state dimension is smaller than the vocabulary size, the
        unembed operation cannot perfectly represent arbitrary logits, since its image
        is restricted to a subspace; this phenomenon is known as the softmax bottleneck
        (cf. https://arxiv.org/abs/1711.03953). Because of this, the inverse can only
        be approximate in general. Here, we use gradient-based optimization to find a
        hidden state that minimizes the KL divergence from the target distribution p to
        unembeded logits q(h): h* = argmin_h KL(p || q(h)).

        Args:
            logits: Tensor of shape `[..., vocab_size]` containing logits to invert.
            h0: Initial guess for the hidden state. If `None`, the least-squares
                solution of the linear equation xU = logits is used, where U is the
                unembedding matrix.
            max_iter: Maximum number of iterations for the optimizer to take.
            optimizer: Optimization algorithm to use. Currently, only "lbfgs" and "sgd"
                are supported.
            prior_weight: The weight of the prior distribution is given in the loss.
            prior: Prior distribution over hidden states used to regularize
                the inversion.
            step_size: The step size for the optimizer.
            tol: Tolerance for the inversion objective.
            weight: Optional tensor of shape `[..., vocab_size]` containing weights
                for each vocabulary item. If `None`, all classes are weighted equally.
        """
        d_model = cast(int, self.unembedding.in_features)
        leading_dims = logits.shape[:-1]

        if h0 is None:
            # Initialize with the Moore-Penrose pseudoinverse
            h0 = torch.zeros((*leading_dims, d_model), device=logits.device)

        # Sanity check the shape of the initial hidden state. Can silently lead to
        # incorrect results due to broadcasting if we don't check this.
        elif h0.shape != (*leading_dims, d_model):
            raise ValueError(
                f"Initial hidden state has shape {h0.shape} but should have shape "
                f"{(*leading_dims, d_model)} given logits shape {logits.shape}."
            )

        h_star = torch.nn.Parameter(h0)
        if optimizer == "lbfgs":
            opt = torch.optim.LBFGS(
                [h_star],
                line_search_fn="strong_wolfe",
                lr=step_size,
                max_iter=max_iter,
                tolerance_change=tol,
            )
        elif optimizer == "sgd":
            opt = torch.optim.SGD([h_star], lr=step_size)
        else:
            raise ValueError(f"Unknown optimizer '{optimizer}'")

        log_p = logits.log_softmax(dim=-1)
        p = log_p.exp()
        if weight is not None:
            p *= weight

        def compute_loss(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            log_q = self(h).log_softmax(-1)
            kl = torch.sum(p * (log_p - log_q), dim=-1).nanmean()
            loss = kl.clone()

            if prior_weight and prior is not None:
                # We evaluate the prior density on the post-norm hidden state,
                # to prevent the pre-norm hidden from collapsing towards zero.
                h_ = self.final_norm(h)
                loss += prior_weight * -prior.log_prob(h_).mean()

            return loss, kl

        nfev = 0  # Number of function evals, like in scipy.optimize.minimize
        loss, kl = log_p.new_tensor(torch.inf), log_p.new_tensor(torch.inf)

        def closure():
            nonlocal nfev, loss, kl
            nfev += 1

            opt.zero_grad(set_to_none=False)
            loss, kl = compute_loss(h_star)

            if not loss.isfinite():
                raise RuntimeError("Inversion objective is not finite.")

            loss.backward()
            return loss

        grad_norm = log_p.new_tensor(torch.inf)
        while nfev < max_iter:
            opt.step(closure)  # type: ignore

            final_grad = h_star.grad
            assert final_grad is not None

            grad_norm = final_grad.norm()
            if grad_norm < tol or loss < tol:
                break

        with torch.no_grad():
            output = InversionOutput(
                preimage=self.final_norm(h_star.data),
                grad_norm=grad_norm,
                kl=kl.detach(),
                loss=loss.detach(),
                nfev=nfev,
            )

        return output
