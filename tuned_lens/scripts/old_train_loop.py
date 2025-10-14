"""Training loop for training a TunedLens model against a transformer on a dataset."""
import dataclasses
import enum
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch as th
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
    """Options of what loss to select when training the model."""

    CE = "ce"
    KL = "kl"

class LensVariant(enum.Enum):
    TUNED = "tuned"
    LORA = "lora"

@dataclass
class State:
    """All of the stateful information in the training loop."""

    dataloader: DataLoader2
    lens: Lens
    opt: Optimizer
    scheduler: LambdaLR
    wandb_id: Optional[str]
    nats_to_bpb: float
    step: int = 0

    def load(self, snapshot_file: Path, device: th.device) -> None:
        """Load a snapshot file."""
        logger.info(f"Loading snapshot from {snapshot_file}...")
        snapshot = th.load(snapshot_file, map_location=device)
        self.step = snapshot["step"]
        self.wandb_id = snapshot["wandb_id"]
        self.lens.load_state_dict(snapshot["lens"])
        self.opt.load_state_dict(snapshot["optim"])
        self.scheduler.load_state_dict(snapshot["scheduler"])
        self.dataloader.load_state_dict(snapshot["dataloader"])

    def save(self, snapshot_file: Path) -> None:
        """Save a snapshot file."""
        logger.info(f"Saving snapshot to {snapshot_file}...")
        if isinstance(self.opt, ZeroRedundancyOptimizer):
            self.opt.consolidate_state_dict()

        th.save(
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
    """Training loop for the tuned lens."""

    model: ing.Model
    """Model configuration."""

    data: ing.Data
    """Data configuration."""

    opt: ing.Optimizer
    """Optimizer configuration."""

    dist: ing.Distributed
    """Configuration for how to distribute the training."""

    output: Path = field(alias=["-o"])
    """Directory to save the lenses to."""

    seed: int = 42
    """Random seed for data shuffling."""

    lens_name_or_path: Optional[str] = field(alias=["-l"], default=None)
    """Name of a pretrained lens to load for fine-tuning."""

    bias_only: Optional[bool] = field(action="store_true")
    """Train only the bias term."""

    num_steps: int = 250
    """Number of training steps."""

    tokens_per_step: int = 2**18
    """Number of tokens per step."""

    wandb: Optional[str] = None
    """Name of run in Weights & Biases."""

    token_shift: Optional[int] = None
    """How to shift the labels wrt the input tokens (1 = next token, 0 = current token,
    -1 = previous token, etc.)"""

    checkpoint_freq: Optional[int] = None
    """Steps between saving a checkpoint. If None, no checkpoints are saved."""

    checkpoint_dir: Optional[Path] = None
    """Directory to save checkpoints to. If None, will use <output>/checkpoints."""

    loss: LossChoice = LossChoice.KL
    """Loss function to use."""

    lens_variant: LensVariant = field(default=LensVariant.TUNED, alias=["--lens-variant"])
    lora_rank: int = field(default=16, alias=["--lora-rank"])

    def __post_init__(self):
        """Set defaults for some fields."""
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.output / "checkpoints"

    def get_lens(self, model: PreTrainedModel) -> Lens:
        """Load or create a lens (TunedLens or LoraLens)."""
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
                # TunedLens: probe is nn.Linear(d,d)
                if isinstance(probe, th.nn.Linear):
                    probe.weight.requires_grad_(False)
                    if probe.bias is not None:
                        probe.bias.requires_grad_(True)
                else:
                    # LoraLens: probe is _LowRankLinear with .down, .up, and optional .bias
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
        """Initialize logging to weights and biases."""
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
        opt: th.optim.Optimizer,
        step: int,
        losses: dict[str, list[float]],
        tuned_lens: TunedLens,
        nats_to_bpb: float,
    ):
        """Log statistics about the training process to weights and biases."""
        if not self.dist.primary or not self.wandb:
            return

        import wandb

        log_dict = {}
        log_dict.update(
            {f"loss/{k}": th.tensor(v).mean() * nats_to_bpb for k, v in losses.items()}
        )

        # Log statistics about optimizer & probes
        for i, probe in enumerate(tuned_lens):
            name = "input" if i == 0 else f"{i - 1}.ffn"
            states = [opt.state[p] for p in probe.parameters()]

            corr = 1 - self.opt.momentum**step
            if self.opt.optimizer == "sgd" and not self.opt.zero:
                log_dict["grad_norm/" + name] = th.cat(
                    [(1 - self.opt.momentum) * s["momentum_buffer"].flatten() / corr
                     for s in states if "momentum_buffer" in s]
                ).norm()
            elif self.opt.optimizer == "adam" and not self.opt.zero:
                log_dict["grad_norm/" + name] = th.cat(
                    [s["exp_avg"].flatten() / corr for s in states if "exp_avg" in s]
                ).norm()

            # Parameter norms by probe type
            if isinstance(probe, th.nn.Linear):
                log_dict["bias_norm/" + name] = probe.bias.data.norm()
                log_dict["weight_norm/" + name] = probe.weight.data.norm()
            else:
                # LoRA-style low-rank: up/down (no internal bias) + optional output bias
                if hasattr(probe, "down"):
                    log_dict["down_weight_norm/" + name] = probe.down.weight.data.norm()
                if hasattr(probe, "up"):
                    log_dict["up_weight_norm/" + name] = probe.up.weight.data.norm()
                if hasattr(probe, "bias") and probe.bias is not None:
                    log_dict["bias_norm/" + name] = probe.bias.data.norm()

        wandb.log(log_dict)

    def snapshot(self, state: State):
        """Save a snapshot of the training process to disk."""
        if self.dist.primary:
            assert self.checkpoint_dir is not None
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            state.save(self.checkpoint_dir / f"snapshot_{state.step}.pth")

    def load_recent_snapshot(self, state: State) -> None:
        """Load the most recent snapshot of the training process from disk."""
        assert self.checkpoint_dir is not None

        if not self.checkpoint_dir.exists():
            logger.warning("No checkpoint directory found. Snapshotting is disabled.")
            return None

        # Find the folder containing the most recent snapshot
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
        """Calculate the number of batches of data to process before taking a step."""
        # chunk_and_tokenize ensures the samples are all the same length
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
            # If the number of samples per step isn't divisible by the global batch
            # size, use ceil division and let the user know about it.
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
        """Initialize the training process."""
        self.dist.init()
        model = tokenizer = data = lens = nats_to_bpb = None

        # Annoyingly, FSDP is incompatible with the `device_map` parameter on
        # `from_pretrained`, because it adds forward hooks to the submodules that move
        # things around to different devices. But `bitsandbytes` requires `device_map`
        # to work at all. So we use `device_map` iff we're using FSDP.
        load_device = self.dist.device if not self.dist.fsdp else None

        if self.dist.primary:
            logger.debug("Primary rank populating cache...")
            model, tokenizer = self.model.load(load_device)
            data, nats_to_bpb = self.data.load(tokenizer)
            lens = self.get_lens(model)

        self.dist.barrier()  # Wait for primary to finish filling the cache

        if not self.dist.primary:
            # Let the non-primary processes load from the cache
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

        # Shard the model using fully shared data parallel
        model = self.dist.shard_model(model)

        self._init_logging(
            model_name=self.model.name, lens=state.lens, wandb_id=state.wandb_id
        )

        tokens_per_sample = len(data[0]["input_ids"])
        grad_acc_steps = self.calculate_gradient_accumulation_steps(
            tokens_per_sample, len(data)
        )

        self.dist.barrier()  # Wait for all processes to finish setup
        logger.info("All processes have completed setup.")
        return state, model, grad_acc_steps

    def execute(self):
        """Trains a TunedLens model against a transformer on a dataset."""
        # Load model, tokenizer, data, and lens
        state, model, grad_acc_steps = self.setup()

        # Enable TF32 for matmuls (safe with bf16)
        if th.cuda.is_available():
            th.backends.cuda.matmul.allow_tf32 = True
            try:
                th.set_float32_matmul_precision("high")
            except AttributeError:
                pass  # older PyTorch

        init_batches = state.step * grad_acc_steps
        total_batches = self.num_steps * grad_acc_steps

        # Wait for all processes to finish setup
        self.dist.barrier()
        logger.info("All processes have completed setup. Starting training.")

        # Main training loop
        t = trange(
            init_batches,
            total_batches,
            desc="Training",
            initial=init_batches,
            total=total_batches,
        )

        # Per-rank lightweight running stats for this optimizer window
        running_loss_sum = None  # tensor on device
        running_loss_count = 0

        # Track peak memory without resetting (primary only)
        last_reported_peak_bytes = 0

        for batch_idx, batch in zip(t, state.dataloader):
            # --- model forward (no autograd at all) ---
            with th.no_grad():
                batch = self.dist.send_to_device(batch)
                output = model(**batch, output_hidden_states=True)

            # Extract and free large containers early
            final_logits = output.logits
            hidden_states = output.hidden_states[:-1]  # list[Tensor]
            del output  # free the container ASAP

            # Prepare labels once
            if self.loss == LossChoice.CE:
                shift = 1 if self.token_shift is None else self.token_shift
                labels = shift_labels(batch["input_ids"], shift)
                del final_logits
            else:
                # Teacher log-probs once (float32 for stability), then free logits
                teacher_logprobs = final_logits.float().log_softmax(dim=-1)
                del final_logits
                teacher_logprobs = teacher_logprobs.to(th.bfloat16)
                shift = 0 if self.token_shift is None else self.token_shift
                labels = shift_labels(teacher_logprobs, shift)
                # Optional memory saver:
                # labels = labels.to(th.bfloat16)

            # --- lens forward + loss ---
            # Backprop PER LAYER to lower peak memory
            for i, h in enumerate(hidden_states):
                with th.autocast(self.dist.device.type, dtype=th.bfloat16):
                    preds = shift_preds(state.lens(h, idx=i), shift)

                    if self.loss == LossChoice.CE:
                        loss_i = th.nn.functional.cross_entropy(
                            preds.flatten(0, -2), labels.flatten()
                        )
                    else:
                        # KL(P||Q) where labels are log P (teacher)
                        logp_pred = preds.log_softmax(-1)
                        loss_i = th.sum(labels.exp() * (labels - logp_pred), dim=-1).mean()

                # Scale for grad accumulation and backprop NOW (frees buffers earlier)
                (loss_i / grad_acc_steps).backward()

                # Lightweight local running sum (every rank keeps one)
                if running_loss_sum is None:
                    running_loss_sum = loss_i.detach().float()
                else:
                    running_loss_sum = running_loss_sum + loss_i.detach().float()
                running_loss_count += 1

            # --- step on boundaries ---
            step, rem = divmod(batch_idx, grad_acc_steps)
            if rem == grad_acc_steps - 1:
                th.nn.utils.clip_grad_norm_(state.lens.parameters(), 1.0)
                state.opt.step()
                state.opt.zero_grad(set_to_none=True)
                state.scheduler.step()

                # Compute per-rank mean, then participate in the all-reduce on ALL ranks
                if running_loss_sum is None:
                    local_mean = th.tensor(0.0, device=self.dist.device)
                else:
                    local_mean = running_loss_sum / max(1, running_loss_count)

                global_mean = maybe_all_reduce(local_mean)  # ALL ranks call this
                mean_loss = float(global_mean.item())

                # Build tqdm postfix (primary only)
                if self.dist.primary:
                    postfix = {"avg_loss": f"{mean_loss * state.nats_to_bpb:.4f}"}

                    if th.cuda.is_available():
                        # peak since process start; no reset needed
                        current_peak_bytes = th.cuda.max_memory_allocated()
                        if current_peak_bytes > last_reported_peak_bytes:
                            last_reported_peak_bytes = current_peak_bytes
                            peak_mem_gb = current_peak_bytes / (1024 ** 3)
                            postfix["peak_mem_GB"] = f"{peak_mem_gb:.2f}"

                    t.set_postfix(postfix)

                    lens = getattr(state.lens, "module", state.lens)
                    self._log(state.opt, step, {"avg": [mean_loss]}, lens, state.nats_to_bpb)

                # reset running stats window
                running_loss_sum = None
                running_loss_count = 0

                state.step = step + 1
                if self.checkpoint_freq and step % self.checkpoint_freq == self.checkpoint_freq - 1:
                    self.snapshot(state)

        if self.dist.primary:
            logger.info(f"Saving lens to {self.output}")
            lens = getattr(state.lens, "module", state.lens)
            lens.save(self.output)