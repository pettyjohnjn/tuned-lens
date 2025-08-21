# train_loop.py
"""Training loop for training a TunedLens model against a transformer on a dataset.

Memory-efficiency highlights
----------------------------
1) Streams hidden states: after the model forward (under no_grad), we immediately
   move *each layer's* hidden state to CPU (optionally pinned, in a reduced dtype).
   During the loss computation we bring back only small (B, tc, d) time-chunks.

2) Teacher targets: **computed chunk-by-chunk**; we never keep full (B, T, K) on GPU.
   Each (B, tc, K) teacher chunk is used immediately for all layers, then freed.

3) KL path computes *subset-only* student logits using idx_subset, avoiding (B, tc, V).

4) Keeps the original public API/CLI surface except for a few extra knobs to control
   offload/streaming behavior (see dataclass fields near the bottom of Train).
"""
import dataclasses
import enum
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
from simple_parsing import field
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torchdata.dataloader2 import DataLoader2
from tqdm.auto import trange
from transformers import PreTrainedModel

import tuned_lens.scripts.ingredients as ing
from tuned_lens import TunedLens, LoraLens
from tuned_lens.utils import maybe_all_reduce, shift_labels, shift_preds
from tuned_lens.nn.lenses import Lens

logger = logging.getLogger(__name__)


class LossChoice(enum.Enum):
    CE = "ce"
    KL = "kl"


class LensVariant(enum.Enum):
    TUNED = "tuned"
    LORA = "lora"


@dataclass
class State:
    dataloader: DataLoader2
    lens: Lens
    opt: Optimizer
    scheduler: LambdaLR
    wandb_id: Optional[str]
    nats_to_bpb: float
    step: int = 0

    def load(self, snapshot_file: Path, device: torch.device) -> None:
        logger.info(f"Loading snapshot from {snapshot_file}...")
        snapshot = torch.load(snapshot_file, map_location=device)
        self.step = snapshot["step"]
        self.wandb_id = snapshot["wandb_id"]
        self.lens.load_state_dict(snapshot["lens"])
        self.opt.load_state_dict(snapshot["optim"])
        self.scheduler.load_state_dict(snapshot["scheduler"])
        self.dataloader.load_state_dict(snapshot["dataloader"])

    def save(self, snapshot_file: Path) -> None:
        logger.info(f"Saving snapshot to {snapshot_file}...")
        if isinstance(self.opt, ZeroRedundancyOptimizer):
            self.opt.consolidate_state_dict()

        torch.save(
            {
                "lens": self.lens.state_dict(),
                "optim": self.opt.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "dataloader": self.dataloader.state_dict(),
                "step": self.step,
                "wandb_id": self.wandb_id,
            },
            snapshot_file,
        )


@dataclass
class Train:
    model: ing.Model
    data: ing.Data
    opt: ing.Optimizer
    dist: ing.Distributed

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

    # ---- memory/efficiency knobs ----
    topk: int = field(default=256, alias=["--topk"])
    time_chunk: int = field(default=256, alias=["--time_chunk"])

    offload_hidden_to_cpu: bool = field(default=True, alias=["--offload-hidden-to-cpu"])
    offload_dtype: str = field(default="bfloat16", alias=["--offload-dtype"])
    pin_memory_offload: bool = field(default=True, alias=["--pin-memory-offload"])

    def __post_init__(self):
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.output / "checkpoints"

    def get_lens(self, model: PreTrainedModel) -> Lens:
        if self.lens_variant == LensVariant.TUNED:
            lens = (
                TunedLens.from_model(model)
                if self.lens_name_or_path is None
                else TunedLens.from_model_and_pretrained(model, self.lens_name_or_path)
            )
        elif self.lens_variant == LensVariant.LORA:
            lens = (
                LoraLens.from_model(model, rank=self.lora_rank)
                if self.lens_name_or_path is None
                else LoraLens.from_model_and_pretrained(model, self.lens_name_or_path)
            )
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
                if isinstance(probe, torch.nn.Linear):
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

    def _get_wandb_id(self) -> Optional[str]:
        if not self.dist.primary or not self.wandb:
            return None
        from wandb.sdk.lib import runid
        return runid.generate_id()

    def _init_logging(self, model_name: str, lens: TunedLens, wandb_id: Optional[str]):
        if not self.dist.primary or not self.wandb:
            return
        logger.debug("Initializing Weights & Biases ...")
        import wandb
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
        opt: torch.optim.Optimizer,
        step: int,
        losses: dict[str, list[float]],
        tuned_lens: TunedLens,
        nats_to_bpb: float,
    ):
        if not self.dist.primary or not self.wandb:
            return
        import wandb
        log_dict = {}
        log_dict.update(
            {f"loss/{k}": torch.tensor(v).mean() * nats_to_bpb for k, v in losses.items()}
        )
        for i, probe in enumerate(tuned_lens):
            name = "input" if i == 0 else f"{i - 1}.ffn"
            states = [opt.state[p] for p in probe.parameters()]
            corr = 1 - self.opt.momentum**step
            if self.opt.optimizer == "sgd" and not self.opt.zero:
                log_dict["grad_norm/" + name] = torch.cat(
                    [
                        (1 - self.opt.momentum) * s["momentum_buffer"].flatten() / corr
                        for s in states
                        if "momentum_buffer" in s
                    ]
                ).norm()
            elif self.opt.optimizer == "adam" and not self.opt.zero:
                log_dict["grad_norm/" + name] = torch.cat(
                    [s["exp_avg"].flatten() / corr for s in states if "exp_avg" in s]
                ).norm()
            if isinstance(probe, torch.nn.Linear):
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

    def snapshot(self, state: State):
        if self.dist.primary:
            assert self.checkpoint_dir is not None
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            state.save(self.checkpoint_dir / f"snapshot_{self.step}.pth")

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

    def setup(self) -> tuple[State, Union[PreTrainedModel, FSDP], int]:
        self.dist.init()
        model = tokenizer = data = lens = nats_to_bpb = None
        load_device = self.dist.device if not self.dist.fsdp else None

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

        assert model and tokenizer and data and lens and nats_to_bpb

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
            model_name=self.model.name, lens=getattr(state.lens, "module", state.lens), wandb_id=state.wandb_id
        )

        tokens_per_sample = len(data[0]["input_ids"])
        grad_acc_steps = self.calculate_gradient_accumulation_steps(
            tokens_per_sample, len(data)
        )

        self.dist.barrier()
        logger.info("All processes have completed setup.")
        return state, model, grad_acc_steps

    # -------- dtype parser ----------
    def _to_dtype(self, name: str) -> torch.dtype:
        name = name.lower()
        if name in ("bf16", "bfloat16"):
            return torch.bfloat16
        if name in ("fp16", "half", "float16"):
            return torch.float16
        if name in ("fp32", "float32"):
            return torch.float32
        raise ValueError(f"Unrecognized dtype string for --offload-dtype: {name}")

    # >>> CHUNKED TOP-K CHANGES: compute teacher only for a single time-chunk.
    @torch.no_grad()
    def _teacher_topk_chunk(
        self,
        final_logits: torch.Tensor,  # (B, T, V) on GPU
        t0: int,
        t1: int,
        shift: int,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return teacher indices/logq for the [t0:t1) slice only.

        Returns:
            idx:  (B, tc, K) long (on CPU, pinned if possible)
            logq: (B, tc, K) bf16 (on CPU, pinned if possible)
        """
        device = final_logits.device
        logits_slice = final_logits[:, t0:t1].float()         # (B, tc, V) on GPU
        vals, idx = logits_slice.topk(k=k, dim=-1)            # (B, tc, K)
        # Apply shift before normalization
        vals = shift_labels(vals, shift).contiguous()
        idx  = shift_labels(idx,  shift).contiguous()
        logq = vals - vals.logsumexp(-1, keepdim=True)        # (B, tc, K)

        # Move to CPU to avoid holding on GPU between layers
        idx_cpu  = idx.to("cpu", dtype=torch.long, non_blocking=True, copy=True)
        logq_cpu = logq.to("cpu", dtype=torch.bfloat16, non_blocking=True, copy=True)

        if self.pin_memory_offload:
            idx_cpu  = idx_cpu.pin_memory()
            logq_cpu = logq_cpu.pin_memory()

        # Cleanup GPU temporaries
        del logits_slice, vals, idx, logq
        return idx_cpu, logq_cpu

    def execute(self):
        state, model, grad_acc_steps = self.setup()

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except AttributeError:
                pass

        init_batches  = state.step * grad_acc_steps
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

        running_loss_sum   = None
        running_loss_count = 0
        last_reported_peak_bytes = 0

        offload_dtype = self._to_dtype(self.offload_dtype)

        for batch_idx, batch in zip(t, state.dataloader):
            # ----- forward pass (no grads) -----
            with torch.no_grad():
                batch = self.dist.send_to_device(batch)
                output = model(**batch, output_hidden_states=True)
                final_logits = output.logits  # (B, T, V)

                # Stream / offload hidden states
                if self.offload_hidden_to_cpu:
                    hidden_storage: list[torch.Tensor] = []
                    for h in output.hidden_states[:-1]:
                        h_cpu = (
                            h.detach()
                            .to("cpu", dtype=offload_dtype, non_blocking=True, copy=True)
                        )
                        if self.pin_memory_offload:
                            h_cpu = h_cpu.pin_memory()
                        hidden_storage.append(h_cpu)
                        del h
                else:
                    hidden_storage = list(output.hidden_states[:-1])
                del output

            # ----- teacher targets setup -----
            if self.loss == LossChoice.CE:
                shift  = 1 if self.token_shift is None else self.token_shift
                labels = shift_labels(batch["input_ids"], shift).to(self.dist.device)
            else:
                shift = 0 if self.token_shift is None else self.token_shift

            # >>> CHUNKED TOP-K CHANGES:
            # time-major loop so each teacher chunk is computed once then reused across all layers
            # Hidden states remain offloaded per layer; we fetch each layer slice per chunk.
            B, T, _ = (hidden_storage[0].shape if hidden_storage else final_logits.shape)

            for t0 in range(0, T, self.time_chunk):
                t1 = min(t0 + self.time_chunk, T)

                # Prepare teacher for this chunk (KL only). Keep on CPU, stream to GPU when used.
                if self.loss == LossChoice.KL:
                    idx_slice_cpu, logq_slice_cpu = self._teacher_topk_chunk(
                        final_logits=final_logits,
                        t0=t0,
                        t1=t1,
                        shift=shift,
                        k=self.topk,
                    )

                # Loop over layers for this time-chunk
                for i, h_store in enumerate(hidden_storage):
                    if self.offload_hidden_to_cpu:
                        h_slice = h_store[:, t0:t1].to(self.dist.device, non_blocking=True)
                    else:
                        h_slice = h_store[:, t0:t1]

                    with torch.autocast(self.dist.device.type, dtype=torch.bfloat16):
                        if self.loss == LossChoice.CE:
                            logits_full = state.lens(h_slice, idx=i)  # (B, tc, V)
                            preds_s     = shift_preds(logits_full, shift)
                            loss_i      = torch.nn.functional.cross_entropy(
                                preds_s.flatten(0, -2),
                                labels[:, t0:t1].flatten(),
                            )
                            del logits_full, preds_s
                        else:
                            # Move teacher chunk to GPU just-in-time
                            idx_slice  = idx_slice_cpu.to(self.dist.device, non_blocking=True)
                            logq_slice = logq_slice_cpu.to(self.dist.device, dtype=torch.bfloat16, non_blocking=True)

                            # Strict subset path — ensure lens never forms (B, tc, V)
                            logits_k = state.lens(
                                h_slice,
                                idx=i,
                                idx_subset=idx_slice,  # (B, tc, K)
                            )
                            preds_s = shift_preds(logits_k, shift)   # (B, tc, K)
                            logp_k  = preds_s.log_softmax(-1)        # (B, tc, K)
                            loss_i  = (logq_slice.exp() * (logq_slice - logp_k)).sum(-1).mean()
                            del logits_k, preds_s, logp_k, idx_slice, logq_slice

                    (loss_i / grad_acc_steps).backward()

                    # running loss
                    if running_loss_sum is None:
                        running_loss_sum = loss_i.detach().float()
                    else:
                        running_loss_sum += loss_i.detach().float()
                    running_loss_count += 1

                    del h_slice  # free per-chunk, per-layer buffer

                # free per-chunk teacher buffers (CPU tensors will be GC'd)
                if self.loss == LossChoice.KL:
                    del idx_slice_cpu, logq_slice_cpu

            # end time-major loop

            # Done with logits entirely
            del final_logits

            # Free hidden state storage after this batch
            if self.offload_hidden_to_cpu:
                for h_store in hidden_storage:
                    del h_store
            del hidden_storage

            # ----- optimizer step on grad-acc boundary -----
            step, rem = divmod(batch_idx, grad_acc_steps)
            if rem == grad_acc_steps - 1:
                torch.nn.utils.clip_grad_norm_(state.lens.parameters(), 1.0)
                state.opt.step()
                state.opt.zero_grad(set_to_none=True)
                state.scheduler.step()

                local_mean  = (
                    running_loss_sum / max(1, running_loss_count)
                    if running_loss_sum is not None
                    else torch.tensor(0.0, device=self.dist.device)
                )
                global_mean = maybe_all_reduce(local_mean)
                mean_loss   = float(global_mean.item())

                if self.dist.primary:
                    postfix = {"avg_loss": f"{mean_loss * state.nats_to_bpb:.4f}"}
                    if torch.cuda.is_available():
                        peak_bytes = torch.cuda.max_memory_allocated()
                        if peak_bytes > last_reported_peak_bytes:
                            last_reported_peak_bytes = peak_bytes
                            postfix["peak_mem_GB"] = f"{peak_bytes / 1_073_741_824:.2f}"
                    t.set_postfix(postfix)
                    self._log(
                        state.opt,
                        step,
                        {"avg": [mean_loss]},
                        getattr(state.lens, "module", state.lens),
                        state.nats_to_bpb,
                    )

                running_loss_sum   = None
                running_loss_count = 0
                state.step         = step + 1

                if (self.checkpoint_freq and
                    step % self.checkpoint_freq == self.checkpoint_freq - 1):
                    self.snapshot(state)

        if self.dist.primary:
            logger.info(f"Saving lens to {self.output}")
            getattr(state.lens, "module", state.lens).save(self.output)