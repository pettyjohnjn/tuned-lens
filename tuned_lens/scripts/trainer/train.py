from __future__ import annotations

import dataclasses
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING

import math

from simple_parsing import field

# Ensure 'ing' is present for runtime type evaluation by simple_parsing
try:  # lightweight; does not pull torch
    import tuned_lens.scripts.ingredients as ing  # noqa: F401
except Exception:
    ing = None  # type: ignore

from .enums import LossChoice, LensVariant
from .state import State
from .constants import GRAD_CLIP_NORM

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # only for static typing
    from transformers import PreTrainedModel
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore


@dataclass
class Train:
    """Training loop for the tuned lens."""

    # Use string annotations to avoid importing types eagerly.
    model: "ing.Model"
    data: "ing.Data"
    opt: "ing.Optimizer"
    dist: "ing.Distributed"

    output: Path = field(alias=["-o"])
    seed: int = 42
    lens_name_or_path: Optional[str] = field(alias=["-l"], default=None)
    bias_only: Optional[bool] = field(action="store_true")
    num_steps: int = 250
    tokens_per_step: int = 2**18
    wandb: Optional[str] = None
    token_shift: Optional[int] = None
    checkpoint_freq: Optional[int] = None
    checkpoint_dir: Optional[Path] = None
    loss: LossChoice = LossChoice.KL
    lens_variant: LensVariant = field(default=LensVariant.TUNED, alias=["--lens-variant"])
    lora_rank: int = field(default=16, alias=["--lora-rank"])

    subset_topk: int = 256
    subset_tail_proxy: bool = True

    def __post_init__(self):
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.output / "checkpoints"

    # ---------- WandB helpers (lazy imports) ----------
    def _get_wandb_id(self) -> Optional[str]:
        if not self.dist.primary or not self.wandb:
            return None
        from wandb.sdk.lib import runid  # lazy

        return runid.generate_id()

    def _init_logging(self, model_name: str, lens, wandb_id: Optional[str]) -> None:
        if not self.dist.primary or not self.wandb:
            return
        import wandb  # lazy

        logger.debug("Initializing Weights & Biases ...")
        wandb.init(
            config=dataclasses.asdict(self),
            group=model_name,
            name=self.wandb,
            id=wandb_id,
            resume="allow",
        )
        wandb.watch(lens)

    def _log(
        self,
        opt,
        step: int,
        losses: dict[str, list[float]],
        tuned_lens,
        nats_to_bpb: float,
    ) -> None:
        if not self.dist.primary or not self.wandb:
            return
        import torch as th  # lazy
        import wandb  # lazy

        log_dict: dict[str, th.Tensor] = {}
        log_dict.update(
            {f"loss/{k}": th.tensor(v).mean() * nats_to_bpb for k, v in losses.items()}
        )

        for i, probe in enumerate(tuned_lens):
            name = "input" if i == 0 else f"{i - 1}.ffn"
            states = [opt.state[p] for p in probe.parameters()]

            corr = 1 - self.opt.momentum**step
            if self.opt.optimizer == "sgd" and not self.opt.zero:
                log_dict["grad_norm/" + name] = th.cat(
                    [
                        (1 - self.opt.momentum) * s["momentum_buffer"].flatten() / corr
                        for s in states
                        if "momentum_buffer" in s
                    ]
                ).norm()
            elif self.opt.optimizer == "adam" and not self.opt.zero:
                log_dict["grad_norm/" + name] = th.cat(
                    [s["exp_avg"].flatten() / corr for s in states if "exp_avg" in s]
                ).norm()

            if isinstance(probe, th.nn.Linear):
                log_dict["bias_norm/" + name] = probe.bias.data.norm()
                log_dict["weight_norm/" + name] = probe.weight.data.norm()
            else:
                if hasattr(probe, "down"):
                    log_dict["down_weight_norm/" + name] = probe.down.weight.data.norm()
                if hasattr(probe, "up"):
                    log_dict["up_weight_norm/" + name] = probe.up.weight.data.norm()
                if hasattr(probe, "bias") and probe.bias is not None:
                    log_dict["bias_norm/" + name] = probe.bias.data.norm()

        wandb.log(log_dict)

    # ---------- Lens factory ----------
    def get_lens(self, model: "PreTrainedModel"):
        import torch as th  # lazy
        from tuned_lens import TunedLens, LoraLens  # lazy

        if self.lens_variant == LensVariant.TUNED:
            if self.lens_name_or_path is None:
                logger.info("Randomly initializing TunedLens...")
                lens = TunedLens.from_model(model)
            else:
                logger.info("Loading pretrained TunedLens...")
                lens = TunedLens.from_model_and_pretrained(model, self.lens_name_or_path)
        elif self.lens_variant == LensVariant.LORA:
            if self.lens_name_or_path is None:
                logger.info(f"Randomly initializing LoraLens (rank={self.lora_rank})...")
                lens = LoraLens.from_model(model, rank=self.lora_rank)
            else:
                logger.info("Loading pretrained LoraLens...")
                lens = LoraLens.from_model_and_pretrained(model, self.lens_name_or_path)
        else:
            raise ValueError(f"Unknown lens_variant {self.lens_variant}")

        dtypes = {p.dtype for p in lens.parameters()}
        assert len(dtypes) == 1, f"Expected all parameters to have the same dtype, got {dtypes}"
        lens_dtype = next(iter(dtypes))
        lens_size = sum(p.numel() * p.element_size() for p in lens.parameters())
        num_bytes = lens_size * (self.opt.per_parameter_optim_state_size() + 1)
        logger.info(f"Lens memory usage: {num_bytes / 2 ** 20:.2f} MB in {lens_dtype}")

        if self.bias_only:
            logger.info("Freezing non-bias parameters (bias-only training).")
            for probe in lens:
                if isinstance(probe, th.nn.Linear):
                    probe.weight.requires_grad_(False)
                    if probe.bias is not None:
                        probe.bias.requires_grad_(True)
                else:
                    if hasattr(probe, "down"):
                        probe.down.weight.requires_grad_(False)
                    if hasattr(probe, "up"):
                        probe.up.weight.requires_grad_(False)
                    if hasattr(probe, "bias") and probe.bias is not None:
                        probe.bias.requires_grad_(True)
        return lens

    # ---------- Checkpointing ----------
    def snapshot(self, state: State) -> None:
        if self.dist.primary:
            assert self.checkpoint_dir is not None
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            state.save(self.checkpoint_dir / f"snapshot_{state.step}.pth")

    def load_recent_snapshot(self, state: State) -> None:
        assert self.checkpoint_dir is not None
        if not self.checkpoint_dir.exists():
            logger.warning("No checkpoint directory found. Snapshotting is disabled.")
            return None

        def sort_key_from_path(p: Path):
            if match := re.match(r".*snapshot_(\d+)\.pth", str(p)):
                return int(match.group(1))
            else:
                return -1

        snapshot_location = max(
            self.checkpoint_dir.glob("snapshot_*.pth"),
            key=sort_key_from_path,
            default=None,
        )
        if snapshot_location is None:
            return None
        state.load(snapshot_location, self.dist.device)

    # ---------- Math ----------
    def calculate_gradient_accumulation_steps(
        self, tokens_per_sample: int, total_samples: int
    ) -> int:
        samples_per_step, rem = divmod(self.tokens_per_step, tokens_per_sample)
        if rem:
            raise ValueError(
                f"Number of tokens per step ({self.tokens_per_step:_}) must be "
                f"divisible by the number of tokens per sample ({tokens_per_sample})."
            )

        if total_samples / samples_per_step < self.num_steps:
            raise ValueError(
                f"Can only take {total_samples / samples_per_step:.2f} steps on "
                f"dataset with --tokens_per_step={self.tokens_per_step}."
                f"Requested {self.num_steps} steps."
            )

        global_batch_size = self.dist.per_gpu_batch_size * self.dist.world_size
        grad_acc_steps, rem = divmod(samples_per_step, global_batch_size)
        if rem:
            grad_acc_steps += 1
            adjusted_count = grad_acc_steps * global_batch_size * tokens_per_sample
            logger.warning(
                f"Note: Increasing grad acc steps from {grad_acc_steps - 1} to "
                f"{grad_acc_steps} to maintain load balance across "
                f"{self.dist.world_size} GPUs."
            )
            logger.warning(
                f"Using {adjusted_count:_} tokens per training step "
                f"({self.tokens_per_step:_} requested)."
            )
        else:
            logger.info(f"Gradient accumulation steps: {grad_acc_steps}")
            logger.info(f"Using {self.tokens_per_step:_} tokens per training step.")
        return grad_acc_steps

    # ---------- Setup and execute ----------
    def setup(self) -> "tuple[State, Union[PreTrainedModel, FSDP], int]":
        self.dist.init()

        load_device = self.dist.device if not self.dist.fsdp else None

        model = tokenizer = data = lens = None
        nats_to_bpb: Optional[float] = None

        if self.dist.primary:
            logger.debug("Primary rank populating cache...")
            model, tokenizer = self.model.load(load_device)
            data, nats_to_bpb = self.data.load(tokenizer)
            lens = self.get_lens(model)

        self.dist.barrier()

        if not self.dist.primary:
            logger.debug("Non-primary rank loading from cache...")
            model, tokenizer = self.model.load(load_device, must_use_cache=True)
            data, nats_to_bpb = self.data.load(tokenizer)
            lens = self.get_lens(model)

        assert model and tokenizer and data and lens and nats_to_bpb is not None

        logger.debug(f"Creating data loader and setting seed to {self.seed} ...")
        dl = self.dist.dataloader(data)
        dl.seed(self.seed)

        logger.debug("Creating optimizer and scheduler ...")
        params = [p for p in lens.parameters() if p.requires_grad]
        opt = self.opt.create_optim(params)
        scheduler = self.opt.create_scheduler(opt, self.num_steps)

        ddp_lens = self.dist.distribute_lens(lens)

        state = State(
            step=0,
            wandb_id=self._get_wandb_id(),
            lens=ddp_lens,  # type: ignore
            opt=opt,
            scheduler=scheduler,
            dataloader=dl,
            nats_to_bpb=nats_to_bpb,
        )

        self.load_recent_snapshot(state)

        model = self.dist.shard_model(model)

        self._init_logging(
            model_name=self.model.name, lens=state.lens, wandb_id=state.wandb_id
        )

        tokens_per_sample = len(data[0]["input_ids"])
        grad_acc_steps = self.calculate_gradient_accumulation_steps(
            tokens_per_sample, len(data)
        )

        self.dist.barrier()
        logger.info("All processes have completed setup.")
        return state, model, grad_acc_steps

    def execute(self) -> None:
        """Trains a TunedLens model against a transformer on a dataset."""
        import torch as th  # lazy
        from tqdm.auto import trange  # lazy
        from tuned_lens.utils import maybe_all_reduce, shift_labels, shift_preds  # lazy

        # ---- Local helpers: seam for future subset-KL ----
        def teacher_stats_from_logits(final_logits: th.Tensor):
            """
            Inputs: final_logits [B, T, V]
            Returns: P[bfloat16], logP[float32], H(P)[scalar]
            """
            logP = final_logits.float().log_softmax(dim=-1)
            P = logP.exp().to(th.bfloat16)
            H = (-(P.float() * logP).sum(dim=-1)).mean()
            return P, logP, H

        def loss_ce(preds: th.Tensor, labels: th.Tensor) -> th.Tensor:
            return th.nn.functional.cross_entropy(preds.flatten(0, -2), labels.flatten())

        def loss_full_kl(preds: th.Tensor, teacher_logprobs: th.Tensor) -> th.Tensor:
            logq = preds.log_softmax(-1)
            return th.sum(teacher_logprobs.exp() * (teacher_logprobs - logq), dim=-1).mean()

        def compute_loss(
            preds: th.Tensor,
            loss_choice: "LossChoice",
            *,
            ce_labels: th.Tensor | None,
            teacher_logprobs: th.Tensor | None,
        ) -> th.Tensor:
            if loss_choice == LossChoice.CE:
                assert ce_labels is not None
                return loss_ce(preds, ce_labels)
            elif loss_choice == LossChoice.KL:
                assert teacher_logprobs is not None
                return loss_full_kl(preds, teacher_logprobs)
            elif loss_choice == LossChoice.SUBSET_KL:
                # raise NotImplementedError("Subset-KL not implemented yet.")
                assert teacher_logprobs is not None
                
                if self.subset_tail_proxy:
                    return topk_with_tail_proxy_kl(preds, teacher_logprobs, self.subset_topk)
            else:
                raise ValueError(f"Unknown loss {loss_choice}")

        def topk_with_tail_proxy_kl(
            preds: th.Tensor,            # [B,T,V] student logits (z)
            teacher_logprobs: th.Tensor, # [B,T,V] log P from teacher
            k: int,
        ) -> th.Tensor:
            """Deterministic Top-K + single tail proxy. Mask-free and numerically stable."""
            assert k > 0
            z = preds.float()                 # keep Stage-B math in fp32
            logP = teacher_logprobs.float()
            B, T, V = z.shape
            k = min(k, V)
            n_tail = V - k
            eps = 1e-20

            # Top-K by teacher probability
            idx = th.topk(logP, k=k, dim=-1, sorted=False).indices          # [B,T,K]
            logP_sub = logP.gather(-1, idx)                                  # [B,T,K]
            P_sub = logP_sub.exp()                                           # [B,T,K]
            z_sub = z.gather(-1, idx)                                        # [B,T,K]

            # Teacher tail mass (exact)
            P_tail = (1.0 - P_sub.sum(dim=-1, keepdim=True)).clamp_min(0.0)  # [B,T,1]

            # Stable log-sum-exp for tail: log(exp(lse_all) - exp(lse_sub))
            lse_all = th.logsumexp(z, dim=-1, keepdim=True)                  # [B,T,1]
            lse_sub = th.logsumexp(z_sub, dim=-1, keepdim=True)              # [B,T,1]
            m = th.maximum(lse_all, lse_sub)                                 # [B,T,1]

            # Compute in higher precision to avoid underflow, then clamp and cast back
            t64 = (lse_all.double() - m.double()).exp() - (lse_sub.double() - m.double()).exp()
            t = t64.clamp_min(1e-300).float()                                # [B,T,1]
            log_tail_sumexp = m + t.log()                                    # [B,T,1]

            # Tail proxy logit
            if n_tail == 0:
                z_tail_proxy = th.full_like(log_tail_sumexp, float("-inf"))
            else:
                z_tail_proxy = log_tail_sumexp - math.log(n_tail)            # [B,T,1]

            # Augment student distribution with tail proxy and compute logQ over K+1 outcomes
            aug_z = th.cat([z_sub, z_tail_proxy], dim=-1)                    # [B,T,K+1]
            logQ_aug = th.log_softmax(aug_z, dim=-1)                         # [B,T,K+1]

            # Augmented teacher probabilities (K exact tokens + exact tail mass)
            logP_tail = P_tail.clamp_min(eps).log()
            P_aug = th.cat([P_sub, P_tail], dim=-1)                          # [B,T,K+1]
            logP_aug = th.cat([logP_sub, logP_tail], dim=-1)                 # [B,T,K+1]

            # Mask tail where P_tail == 0
            tail_mask = th.cat([th.ones_like(P_sub, dtype=th.bool), (P_tail > 0)], dim=-1)
            P_aug = th.where(tail_mask, P_aug, th.zeros_like(P_aug))
            logP_aug = th.where(tail_mask, logP_aug, th.zeros_like(logP_aug))

            # KL = E_{P_aug}[log P_aug - log Q_aug]
            kl = (P_aug * (logP_aug - logQ_aug)).sum(dim=-1).mean()

            # Fallback: pure Top-K truncated KL if non-finite (should not trigger)
            if not th.isfinite(kl):
                logP_sub_n = logP_sub - th.logsumexp(logP_sub, dim=-1, keepdim=True)
                P_sub_n = logP_sub_n.exp()
                logQ_sub_n = th.log_softmax(z_sub, dim=-1)
                kl = (P_sub_n * (logP_sub_n - logQ_sub_n)).sum(dim=-1).mean()
            return kl

        state, model, grad_acc_steps = self.setup()

        if th.cuda.is_available():
            th.backends.cuda.matmul.allow_tf32 = True
            try:
                th.set_float32_matmul_precision("high")
            except AttributeError:
                pass

        init_batches = state.step * grad_acc_steps
        total_batches = self.num_steps * grad_acc_steps

        self.dist.barrier()
        logger.info("All processes have completed setup. Starting training.")

        t = trange(
            init_batches,
            total_batches,
            desc="Training",
            initial=init_batches,
            total=total_batches,
        )

        running_loss_sum = None
        running_loss_count = 0
        last_reported_peak_bytes = 0

        for batch_idx, batch in zip(t, state.dataloader):
            with th.no_grad():
                batch = self.dist.send_to_device(batch)
                output = model(**batch, output_hidden_states=True)

            final_logits = output.logits
            hidden_states = output.hidden_states[:-1]
            del output

            # Unify teacher prep and label shifting
            if self.loss == LossChoice.CE:
                shift = 1 if self.token_shift is None else self.token_shift
                ce_labels = shift_labels(batch["input_ids"], shift)
                teacher_logprobs = None
            else:
                teacher_logprobs = final_logits.float().log_softmax(dim=-1)
                shift = 0 if self.token_shift is None else self.token_shift
                teacher_logprobs = shift_labels(teacher_logprobs, shift)
                ce_labels = None
            del final_logits

            for i, h in enumerate(hidden_states):
                with th.autocast(self.dist.device.type, dtype=th.bfloat16):
                    preds = shift_preds(state.lens(h, idx=i), shift)
                    loss_i = compute_loss(
                        preds,
                        self.loss,
                        ce_labels=ce_labels,
                        teacher_logprobs=teacher_logprobs,
                    )

                (loss_i / grad_acc_steps).backward()

                if running_loss_sum is None:
                    running_loss_sum = loss_i.detach().float()
                else:
                    running_loss_sum = running_loss_sum + loss_i.detach().float()
                running_loss_count += 1

            step, rem = divmod(batch_idx, grad_acc_steps)
            if rem == grad_acc_steps - 1:
                th.nn.utils.clip_grad_norm_(state.lens.parameters(), GRAD_CLIP_NORM)
                state.opt.step()
                state.opt.zero_grad(set_to_none=True)
                state.scheduler.step()

                local_mean = (
                    th.tensor(0.0, device=self.dist.device)
                    if running_loss_sum is None
                    else running_loss_sum / max(1, running_loss_count)
                )
                global_mean = maybe_all_reduce(local_mean)
                mean_loss = float(global_mean.item())

                if self.dist.primary:
                    postfix = {"avg_loss": f"{mean_loss * state.nats_to_bpb:.4f}"}
                    if th.cuda.is_available():
                        current_peak_bytes = th.cuda.max_memory_allocated()
                        if current_peak_bytes > last_reported_peak_bytes:
                            last_reported_peak_bytes = current_peak_bytes
                            peak_mem_gb = current_peak_bytes / (1024 ** 3)
                            postfix["peak_mem_GB"] = f"{peak_mem_gb:.2f}"
                    t.set_postfix(postfix)

                    lens_mod = getattr(state.lens, "module", state.lens)
                    self._log(state.opt, step, {"avg": [mean_loss]}, lens_mod, state.nats_to_bpb)

                running_loss_sum = None
                running_loss_count = 0

                state.step = step + 1
                if self.checkpoint_freq and step % self.checkpoint_freq == self.checkpoint_freq - 1:
                    self.snapshot(state)

        if self.dist.primary:
            logger.info(f"Saving lens to {self.output}")
            lens_mod = getattr(state.lens, "module", state.lens)
            lens_mod.save(self.output)


__all__ = ["Train"]