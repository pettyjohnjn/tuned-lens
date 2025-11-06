from __future__ import annotations

import dataclasses
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING

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

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore


@dataclass
class Train:
    """Training loop for the tuned lens."""

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

    # Stage A config
    subset_topk: int = 256  # head size (K_head)

    # Head–tail subset KL knobs
    tail_k: int = 64                 # samples from tail per (B,T)
    tail_clip: float = 50.0          # cap on importance ratios
    self_normalize_tail: bool = True
    tail_proposal: str = field(default="uniform", alias=["--tail-proposal"])  # "uniform"|"teacher"
    tail_oversample: int = field(default=4, alias=["--tail-oversample"])      # candidate multiplier M=k*oversample

    def __post_init__(self):
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.output / "checkpoints"

    # ---------- WandB helpers ----------
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
                import torch as th
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

        # ---- helpers ----
        def loss_ce(preds: th.Tensor, labels: th.Tensor) -> th.Tensor:
            return th.nn.functional.cross_entropy(preds.flatten(0, -2), labels.flatten())

        def loss_full_kl(preds: th.Tensor, teacher_logprobs: th.Tensor) -> th.Tensor:
            logq = preds.log_softmax(-1)
            return th.sum(teacher_logprobs.exp() * (teacher_logprobs - logq), dim=-1).mean()

        def _lens_forward_subset(h_shifted: th.Tensor, layer_idx: int, idx: th.Tensor) -> th.Tensor:
            lens_mod = getattr(state.lens, "module", state.lens)  # unwrap DDP/FSDP
            return lens_mod.forward_subset(h_shifted, layer_idx, idx)

        # ---- head–tail subset KL (deterministic head + sampled tail) ----
        def _gather_topk(logP: th.Tensor, k: int):
            k = min(k, logP.size(-1))
            idx = th.topk(logP, k=k, dim=-1, sorted=False).indices  # [B,T,k]
            logP_sub = logP.gather(-1, idx)                         # [B,T,k]
            return idx, logP_sub

        def _sample_tail_uniform_excluding(idx_head: th.Tensor, vocab_size: int, k: int) -> th.Tensor:
            """
            Uniform sampling from tail = vocab \ head. Vectorized fast path with a rare slow fallback.
            idx_head: [B,T,K_h] -> returns [B,T,k]
            """
            B, T, K = idx_head.shape
            V = vocab_size
            k = min(k, max(1, V - K))

            def draw(kdraw: int) -> th.Tensor:
                return th.randint(low=0, high=V, size=(B, T, kdraw), device=idx_head.device)

            cand = draw(k * 2)
            neq = cand.unsqueeze(-1) != idx_head.unsqueeze(-2)  # [B,T,2k,K_h]
            not_head = neq.all(dim=-1)                          # [B,T,2k]
            cand_masked = th.where(not_head, cand, th.full_like(cand, -1))
            valid = (cand_masked >= 0).int()
            topk_valid = th.topk(valid, k=k, dim=-1).indices
            tail_idx = cand_masked.gather(-1, topk_valid)
            if (tail_idx < 0).any():
                # slow fallback
                tail_idx_list = []
                for b in range(B):
                    row = []
                    for t in range(T):
                        head_set = set(idx_head[b, t].tolist())
                        out = []
                        while len(out) < k:
                            x = int(th.randint(0, V, ()).item())
                            if x not in head_set:
                                out.append(x)
                        row.append(out)
                    tail_idx_list.append(row)
                tail_idx = th.tensor(tail_idx_list, device=idx_head.device, dtype=th.long)
            return tail_idx

        def _sample_tail_teacher_biased(
            idx_head: th.Tensor,
            logP: th.Tensor,      # [B,T,V] teacher log-probs
            k: int,               # k_tail
            oversample: int,      # candidate multiplier
        ) -> tuple[th.Tensor, th.Tensor]:
            """
            Returns:
              tail_idx: [B,T,k] vocab indices from tail
              q_sel:    [B,T,k] proposal probabilities for selected indices
            Proposal: sample a small uniform candidate set from tail, then
            sample with replacement from that set proportional to teacher probs.
            """
            import torch as th
            B, T, K_h = idx_head.shape
            V = logP.size(-1)
            tail_size = max(0, V - K_h)
            if tail_size == 0:
                # Degenerate case: no tail
                empty = th.zeros(B, T, k, device=idx_head.device, dtype=th.long)
                q_zero = th.full_like(empty, 1.0 / max(1, k), dtype=logP.dtype)
                return empty, q_zero

            M = min(max(k, 1) * max(1, oversample), tail_size)
            cand = _sample_tail_uniform_excluding(idx_head, V, M)       # [B,T,M]

            logP_C = logP.gather(-1, cand).float()                      # [B,T,M]
            logZ_C = th.logsumexp(logP_C, dim=-1, keepdim=True)         # [B,T,1]
            q_C = th.exp(logP_C - logZ_C).clamp_min(1e-12)              # [B,T,M]

            q_flat = q_C.reshape(B * T, M)
            sel_in_c = th.multinomial(q_flat, num_samples=k, replacement=True)  # [B*T,k]
            sel_in_c = sel_in_c.view(B, T, k)

            tail_idx = cand.gather(-1, sel_in_c)                        # [B,T,k]
            q_sel = q_C.gather(-1, sel_in_c)                            # [B,T,k]
            return tail_idx, q_sel

        def head_tail_subset_kl(
            h_shifted: th.Tensor,
            layer_idx: int,
            teacher_logprobs: th.Tensor,
            k_head: int,
            k_tail: int,
            tail_clip: float,
            self_norm: bool,
        ) -> th.Tensor:
            """
            Head: exact subset KL on top-k_head (renormalized within head).
            Tail: self-normalized importance-sampled subset KL on k_tail with selectable proposal.
            """
            logP = teacher_logprobs.float()                # [B,T,V]
            V = logP.size(-1)

            # Head deterministic term
            idxH, logP_H = _gather_topk(logP, k_head)      # [B,T,K_h], [B,T,K_h]
            z_H = _lens_forward_subset(h_shifted, layer_idx, idxH).float()  # [B,T,K_h]
            logP_Hn = logP_H - th.logsumexp(logP_H, dim=-1, keepdim=True)
            P_Hn = logP_Hn.exp()
            logQ_Hn = th.log_softmax(z_H, dim=-1)
            head_kl = th.sum(P_Hn * (logP_Hn - logQ_Hn), dim=-1)            # [B,T]

            # Tail sampled term
            if self.tail_proposal == "teacher":
                idxT, q_sel = _sample_tail_teacher_biased(idxH, logP, k_tail, self.tail_oversample)
                logP_T = logP.gather(-1, idxT)                              # [B,T,K_t]
                z_T = _lens_forward_subset(h_shifted, layer_idx, idxT).float()
                w = (logP_T.exp() / q_sel).clamp_min(1e-12)                 # IS ratios under teacher-biased q
            elif self.tail_proposal == "uniform":
                idxT = _sample_tail_uniform_excluding(idxH, V, k_tail)      # [B,T,K_t]
                logP_T = logP.gather(-1, idxT)
                z_T = _lens_forward_subset(h_shifted, layer_idx, idxT).float()
                tail_size = max(1, V - idxH.size(-1))
                q = 1.0 / tail_size                                         # uniform over tail
                w = (logP_T.exp() / q)
            else:
                raise ValueError(f"Unknown tail_proposal {self.tail_proposal}")

            if tail_clip is not None and tail_clip > 0:
                w = th.clamp(w, max=tail_clip)

            if self_norm:
                w_sum = w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                w_sn = w / w_sum
            else:
                w_sn = w / float(k_tail)

            logP_Tn = logP_T - th.logsumexp(logP_T, dim=-1, keepdim=True)
            logQ_Tn = th.log_softmax(z_T, dim=-1)
            tail_term = th.sum(w_sn * (logP_Tn - logQ_Tn), dim=-1)           # [B,T]

            loss_bt = head_kl + tail_term
            return loss_bt.mean()

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
                # handled in per-layer branch
                raise RuntimeError("SUBSET_KL handled in per-layer branch.")
            else:
                raise ValueError(f"Unknown loss {loss_choice}")

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

            if self.loss == LossChoice.CE:
                shift = 1 if self.token_shift is None else self.token_shift
                ce_labels = shift_labels(batch["input_ids"], shift)
                teacher_logprobs = None
                del final_logits
            else:
                teacher_logprobs = final_logits.float().log_softmax(dim=-1).to(th.bfloat16)
                shift = 0 if self.token_shift is None else self.token_shift
                teacher_logprobs = shift_labels(teacher_logprobs, shift)
                ce_labels = None
                del final_logits

            for i, h in enumerate(hidden_states):
                with th.autocast(self.dist.device.type, dtype=th.bfloat16):
                    if self.loss == LossChoice.SUBSET_KL:
                        h_shift = shift_preds(h, shift)
                        loss_i = head_tail_subset_kl(
                            h_shift,
                            i,
                            teacher_logprobs,
                            k_head=self.subset_topk,
                            k_tail=self.tail_k,
                            tail_clip=self.tail_clip,
                            self_norm=self.self_normalize_tail,
                        )
                    else:
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