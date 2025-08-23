"""Evaluation loop for logit and multi-hook tuned lenses (residual_out/attn_out/mlp_out)."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Dict, List, Literal, Optional

import torch as th
from simple_parsing import field
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from tuned_lens.nn.lenses import Lens, LogitLens, TunedLens, LoraLens
from tuned_lens.scripts.ingredients import Data, Distributed, Model
from tuned_lens.stats import LogitStats
from tuned_lens.utils import maybe_all_reduce, pytree_map, pytree_stack, shift_labels, shift_preds

# c_proj hook capture for attn/mlp (matches training)
from tuned_lens.scripts.hook_points import HookManager

LensType = Literal["logit", "tuned"]

logger = logging.getLogger(__name__)


def _nested_dict():
    return defaultdict(_nested_dict)


@dataclass
class Eval:
    """CLI args for evaluation."""

    data: Data
    model: Model
    dist: Distributed

    output: Path = field(alias=["-o"])
    """Folder to save the eval results to."""

    lens_name: Optional[str] = field(alias=["-l"], default=None)
    """
    Path to tuned lens root. For multi-hook training, this should be the
    parent folder containing subfolders: residual_out/, attn_out/, mlp_out/.
    For legacy single-lens runs, you can point directly at that lens folder.
    """

    logit: bool = True
    """Whether to evaluate the logit lens."""

    seed: int = 42
    """Random seed used for data shuffling."""

    tokens: Optional[int] = None
    """Number of tokens to evaluate on. If None, will use the entire dataset."""

    token_shift: int = field(default=1)
    """How to shift labels wrt the input tokens (1 = next token)."""

    per_gpu_batch_size: int = 1
    """Samples per GPU."""

    layer_transfer: bool = field(action="store_true")
    """Evaluate how a lens transfers across layers (NxN)."""

    record_logit_stats: bool = field(action="store_true")
    """Record marginal token distribution stats per layer."""

    # Which hook points to evaluate
    hook_points: List[str] = field(
        default_factory=lambda: ["residual_out"],
        alias=["--hook_points", "--hook_point"],  # accept plural or singular
        help="One or more of: residual_out attn_out mlp_out",
    )

    # --------- speed helpers (optional) ----------
    max_batches: Optional[int] = field(default=None, alias=["--max_batches"])
    """Stop after this many batches regardless of --tokens."""

    layers: Optional[str] = field(default=None, alias=["--layers"])
    """Comma/range list like '0,5,10-12'. If set, only evaluate these layers."""

    layer_stride: int = field(default=1, alias=["--layer_stride"])
    """Evaluate every Nth layer (applied after --layers filter)."""

    # --------------- lens loading helpers ---------------

    def _load_single_tuned_lens(self, model: PreTrainedModel, path: Path) -> Lens:
        """
        Load either a TunedLens or a LoraLens from `path`, matching whatever was saved.
        Prefers TunedLens; falls back to LoraLens if state dict has LoRA keys.
        """
        # First attempt: TunedLens
        try:
            return TunedLens.from_model_and_pretrained(model, str(path))
        except RuntimeError as e:
            msg = str(e)
            # Heuristic: LoRA checkpoints contain ".down.weight"/".up.weight"
            if (".down.weight" in msg) or (".up.weight" in msg) or ("Unexpected key(s)" in msg and "down.weight" in msg):
                # Retry as LoraLens
                return LoraLens.from_model_and_pretrained(model, str(path))
            raise  # re-raise if it's some other issue

    def _load_hook_lenses(self, model: PreTrainedModel) -> Dict[str, Lens]:
        """
        Returns a dict like:
            {
              "logit": LogitLens(...),             # optional
              "residual_out": TunedLens(...),      # if requested & found
              "attn_out": TunedLens(...),          # if requested & found
              "mlp_out": TunedLens(...),           # if requested & found
            }
        """
        lenses: Dict[str, Lens] = {}

        if self.logit:
            lenses["logit"] = LogitLens.from_model(model)

        if self.lens_name is None:
            # No tuned lens provided; only logit (if requested)
            return lenses

        root = Path(self.lens_name)

        # If the user passed a multi-hook root folder, look for subfolders per hook
        subdirs = {hp: root / hp for hp in ("residual_out", "attn_out", "mlp_out")}
        has_subdirs = any(sd.exists() and sd.is_dir() for sd in subdirs.values())

        if has_subdirs:
            for hp in self.hook_points:
                sd = subdirs.get(hp)
                if hp in ("attn_out", "mlp_out", "residual_out") and sd and sd.exists():
                    lenses[hp] = self._load_single_tuned_lens(model, sd)
                else:
                    logger.warning(f"Requested hook '{hp}' but no lens found at {sd}. Skipping.")
        else:
            # Legacy: a single tuned lens folder was provided; treat it as residual_out unless user asked otherwise
            if "residual_out" in self.hook_points:
                lenses["residual_out"] = self._load_single_tuned_lens(model, root)
            else:
                logger.warning(
                    f"Single lens path provided ({root}) but requested non-residual hooks {self.hook_points}. "
                    "Attempting to load lens for those hooks as well."
                )
                for hp in self.hook_points:
                    lenses[hp] = self._load_single_tuned_lens(model, root)

        return lenses

    def load_lenses(self, model: PreTrainedModel) -> Dict[str, Lens]:
        """Load the requested lenses."""
        return self._load_hook_lenses(model)

    # --------------- batching helpers ---------------

    def calculate_batch_limit(self, tokens_per_sample: int):
        """Calculate total number of batches to evaluate on."""
        assert self.tokens is not None
        global_batch_size = self.dist.world_size * self.per_gpu_batch_size
        tokens_per_batch = global_batch_size * tokens_per_sample
        return self.tokens // tokens_per_batch

    # --------------- layer filtering ---------------

    def _parse_layers(self, L: int) -> List[int]:
        """Return a sorted list of layer indices to evaluate."""
        stride = max(1, int(self.layer_stride))
        if not self.layers:
            return list(range(0, L, stride))
        keep = set()
        for part in self.layers.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                keep.update(range(int(a), int(b) + 1))
            else:
                keep.add(int(part))
        return sorted(i for i in keep if 0 <= i < L)[::stride]

    # --------------- logit stats bookkeeping ---------------

    def _initialize_logit_stats_recorders(self, lenses: dict[str, Lens], total_layers: int):
        if self.record_logit_stats:
            self.logit_stats_recorders = {
                lens_name: {f"layer_{i}": LogitStats() for i in range(total_layers)}
                for lens_name in lenses.keys()
            }
            self.logit_stats_recorder_final = LogitStats()
        else:
            self.logit_stats_recorders = None
            self.logit_stats_recorder_final = None

    def _record_logit_stats(self, logp: th.Tensor, layer: int, lens_name: str):
        if self.logit_stats_recorders is not None:
            self.logit_stats_recorders[lens_name][f"layer_{layer}"].update(
                logp, assume_normalized=True
            )

    def _record_logit_stats_final(self, logp: th.Tensor):
        if self.logit_stats_recorder_final is not None:
            self.logit_stats_recorder_final.update(logp, assume_normalized=True)

    def _save_logit_stats(self) -> defaultdict:
        logit_stats = _nested_dict()
        if self.logit_stats_recorders is not None:
            for lens_name, recorders in self.logit_stats_recorders.items():
                for layer, recorder in recorders.items():
                    recorder.all_reduce_()
                    logit_stats[lens_name]["logit_stats"][layer] = (
                        recorder.marginal_probs.cpu().numpy().tolist()
                    )

        if self.logit_stats_recorder_final is not None:
            self.logit_stats_recorder_final.all_reduce_()
            logit_stats["baseline"]["logit_stats"]["final"] = (
                self.logit_stats_recorder_final.marginal_probs.cpu().numpy().tolist()
            )
        return logit_stats

    # --------------- core eval over a single hook storage ---------------

    def _eval_on_storage(
        self,
        lens: Lens,
        storage: List[th.Tensor],
        final_probs: th.Tensor,
        final_lps: th.Tensor,
        labels: th.Tensor,
        lens_key: str,
        batch_output: defaultdict,
        total_layers: int,
        keep_layers: List[int],
    ):
        """
        Evaluate a given lens over a per-layer storage list.
        storage[i] shape: (B, T, d)
        keep_layers lists the *global* layer indices corresponding to storage ordering.
        """
        for local_idx, h in enumerate(storage):
            layer_idx = keep_layers[local_idx]
            layer_name = f"layer_{layer_idx}"

            lens_lps = lens(h, idx=layer_idx).log_softmax(dim=-1)
            lens_probs = lens_lps.exp()

            # stats
            self._record_logit_stats(lens_lps, layer_idx, lens_key)

            # CE
            batch_output[lens_key]["ce"][layer_name] = th.nn.functional.cross_entropy(
                shift_preds(lens_lps, self.token_shift).flatten(0, 1),
                labels.flatten(),
                reduction="none",
            )
            # entropy
            batch_output[lens_key]["entropy"][layer_name] = th.sum(-lens_probs * lens_lps, dim=-1)
            # KL (teacher: final; student: lens)
            batch_output[lens_key]["kl"][layer_name] = th.sum(final_probs * (final_lps - lens_lps), dim=-1)

            # optional transfer: feed the same h into all layer-specific probes
            if self.layer_transfer:
                for i in keep_layers:
                    trans_name = f"layer_{i}"
                    transfer_lps = lens(h, idx=i).log_softmax(dim=-1)
                    batch_output[lens_key]["layer_transfer"]["ce"][trans_name][layer_name] = (
                        th.nn.functional.cross_entropy(
                            shift_preds(transfer_lps, self.token_shift).flatten(0, 1),
                            labels.flatten(),
                        )
                    )
                    batch_output[lens_key]["layer_transfer"]["kl"][trans_name][layer_name] = (
                        th.sum(lens_probs * (lens_lps - transfer_lps), dim=-1).mean()
                    )

    # --------------- main entrypoint ---------------

    @th.autocast("cuda", enabled=th.cuda.is_available())
    @th.no_grad()
    def execute(self):
        """Evaluates (multi-hook) TunedLens/LogitLens against a transformer on a dataset."""
        # init & load
        self.dist.init()
        model = tokenizer = data = lenses = nats_to_bpb = None

        load_device = self.dist.device if not self.dist.fsdp else None
        if self.dist.primary:
            model, tokenizer = self.model.load(load_device)
            data, nats_to_bpb = self.data.load(tokenizer)
            lenses = self.load_lenses(model)

        self.dist.barrier()

        if not self.dist.primary:
            model, tokenizer = self.model.load(load_device, must_use_cache=True)
            data, nats_to_bpb = self.data.load(tokenizer)
            lenses = self.load_lenses(model)

        assert model and tokenizer and data and lenses and nats_to_bpb

        model = self.dist.shard_model(model)

        # move lenses to device & eval
        lenses = {k: v.to(self.dist.device) for k, v in lenses.items()}
        for lens in lenses.values():
            lens.eval()

        dl = self.dist.dataloader(data)
        dl.seed(self.seed)

        # token limit handling
        if self.tokens is not None:
            tokens_per_sample = len(data[0]["input_ids"])
            if self.tokens > len(data) * tokens_per_sample:
                raise ValueError(
                    f"Requested {self.tokens} tokens, but dataset has {len(data) * tokens_per_sample}."
                )
            batch_limit = self.calculate_batch_limit(tokens_per_sample)
            assert batch_limit > 0, "Batch limit must be positive."
            dl = islice(dl, batch_limit)
            total = batch_limit
        else:
            total = len(data) // self.dist.world_size

        # additional hard cap on batches
        if self.max_batches is not None:
            dl = islice(dl, self.max_batches)
            total = self.max_batches

        L = model.config.num_hidden_layers
        keep_layers = self._parse_layers(L)  # which global layers to evaluate

        self._initialize_logit_stats_recorders(lenses, L)

        root_dir = self.output
        root_dir.mkdir(exist_ok=True, parents=True)

        batches = []

        self.dist.barrier()
        logger.info(f"All processes initialized. Evaluating {total} batches on layers {keep_layers}.")

        # Prepare hook manager for c_proj outputs if needed
        need_attn = "attn_out" in self.hook_points and "attn_out" in lenses
        need_mlp  = "mlp_out" in self.hook_points and "mlp_out" in lenses

        hook_mgr = HookManager(
            model=model.module if hasattr(model, "module") else model,
            hook_points=[hp for hp in self.hook_points if hp in ("attn_out", "mlp_out")],
            offload_to_cpu=False,             # eval: keep on device if possible
            offload_dtype=th.bfloat16,        # ignored when offload_to_cpu=False
            pin_memory=False,
        )
        if need_attn or need_mlp:
            hook_mgr.register()

        pbar = tqdm(dl, desc="Evaluating", position=self.dist.rank, total=total)
        try:
            for batch in pbar:
                batch = self.dist.send_to_device(batch)

                # enable residual_out capture if requested
                output = model(**batch, output_hidden_states=("residual_out" in self.hook_points))
                final_lps = output.logits.log_softmax(dim=-1)
                final_probs = final_lps.exp()
                assert not th.isnan(output.logits).any(), "Logits are NaN"

                labels = shift_labels(batch["input_ids"], self.token_shift)

                # build storages per hook, filtered to keep_layers
                residual_storage: List[th.Tensor] = []
                if "residual_out" in self.hook_points:
                    # output.hidden_states includes input embedding at 0 and final at -1; we want per-layer outputs
                    # original code used hidden_states[:-1]; we now select specific layers
                    all_hidden = list(output.hidden_states[:-1])
                    residual_storage = [all_hidden[i] for i in keep_layers]

                # Hook storages (attn/mlp) come back ordered by layer; filter by keep_layers index
                attn_storage: List[th.Tensor] = []
                if need_attn:
                    full = hook_mgr.get_buffers("attn_out")
                    attn_storage = [full[i] for i in keep_layers]

                mlp_storage: List[th.Tensor] = []
                if need_mlp:
                    full = hook_mgr.get_buffers("mlp_out")
                    mlp_storage = [full[i] for i in keep_layers]

                batch_output = _nested_dict()

                # ---- evaluate logit lens (acts on final logits directly) ----
                if "logit" in lenses:
                    # mirror baseline naming pattern
                    batch_output["logit"]["ce"]["final"] = th.nn.functional.cross_entropy(
                        shift_preds(final_lps, self.token_shift).flatten(0, 1),
                        labels.flatten(),
                        reduction="none",
                    )
                    batch_output["logit"]["entropy"]["final"] = th.sum(-final_probs * final_lps, dim=-1)
                    batch_output["logit"]["kl"]["final"] = th.zeros_like(
                        batch_output["logit"]["entropy"]["final"]
                    )  # KL(final||final)=0

                # ---- evaluate tuned lenses per hook ----
                if "residual_out" in lenses and residual_storage:
                    self._eval_on_storage(
                        lens=lenses["residual_out"],
                        storage=residual_storage,
                        final_probs=final_probs,
                        final_lps=final_lps,
                        labels=labels,
                        lens_key="residual_out",
                        batch_output=batch_output,
                        total_layers=L,
                        keep_layers=keep_layers,
                    )

                if "attn_out" in lenses and attn_storage:
                    self._eval_on_storage(
                        lens=lenses["attn_out"],
                        storage=attn_storage,
                        final_probs=final_probs,
                        final_lps=final_lps,
                        labels=labels,
                        lens_key="attn_out",
                        batch_output=batch_output,
                        total_layers=L,
                        keep_layers=keep_layers,
                    )

                if "mlp_out" in lenses and mlp_storage:
                    self._eval_on_storage(
                        lens=lenses["mlp_out"],
                        storage=mlp_storage,
                        final_probs=final_probs,
                        final_lps=final_lps,
                        labels=labels,
                        lens_key="mlp_out",
                        batch_output=batch_output,
                        total_layers=L,
                        keep_layers=keep_layers,
                    )

                # ---- baseline (final layer) ----
                batch_output["baseline"]["ce"]["final"] = th.nn.functional.cross_entropy(
                    shift_preds(final_lps, self.token_shift).flatten(0, 1),
                    labels.flatten(),
                    reduction="none",
                )
                batch_output["baseline"]["entropy"]["final"] = th.sum(-final_probs * final_lps, dim=-1)

                batches.append(pytree_map(th.mean, batch_output))  # type: ignore[arg-type]

                self._record_logit_stats_final(final_lps)

                # clear hook buffers between batches
                hook_mgr.clear()
        finally:
            hook_mgr.remove()

        pbar.close()

        # aggregate across batches & workers
        agg = pytree_map(lambda x: nats_to_bpb * x.mean(), pytree_stack(batches))
        agg = pytree_map(lambda x: maybe_all_reduce(x), agg)
        agg = pytree_map(lambda x: x.cpu().numpy().item(), agg)
        assert isinstance(agg, dict)

        batches = pytree_map(lambda x: nats_to_bpb * x, batches)
        batches = pytree_map(lambda x: maybe_all_reduce(x), batches)
        batches = pytree_map(lambda x: x.cpu().item(), batches)
        assert isinstance(batches, list)

        logit_stats = self._save_logit_stats()

        if self.dist.primary:
            with (root_dir / "batches.jsonl").open("w") as f:
                json.dump(batches, f)
            with (root_dir / "aggregate_metrics.json").open("w") as f:
                json.dump(agg, f)
            if self.record_logit_stats:
                with (root_dir / "logit_stats.json").open("w") as f:
                    json.dump(logit_stats, f)