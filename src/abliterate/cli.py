"""Command-line interface.

    abliterate run       --model ... [--lora ...] [--select auto]   # full pipeline
    abliterate directions --model ... -o out/directions.pt          # just compute
    abliterate apply     --model ... --directions out/directions.pt # orthogonalize
    abliterate evaluate  --model ... [--directions ... ]            # refusal rate

Either pass a YAML via --config, or drive everything with flags (flags override
config values when both are given).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from .config import AblationConfig, Config, DataConfig, ModelConfig


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="YAML config path (flags override its values)")
    p.add_argument("--model", help="HF model id or path (base or full model)")
    p.add_argument("--lora", help="PEFT/LoRA adapter dir to merge (e.g. reforge output)")
    p.add_argument("--tokenizer", help="tokenizer path (defaults to --model)")
    p.add_argument("--dtype", default=None, choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device-map", default=None, help="device_map for loading (default: auto)")
    p.add_argument("--trust-remote-code", action="store_true", default=None)


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--harmful", help="refusal-eliciting prompts: file or dataset name")
    p.add_argument("--harmless", help="benign prompts: file or dataset name")
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)


def _add_ablation_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--layer-index", type=int, default=None, help="explicit direction index")
    p.add_argument("--layer-fraction", type=float, default=None, help="fractional depth (0-1)")
    p.add_argument("--position", type=int, default=None, help="token position for activations")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--no-embed", dest="ablate_embed", action="store_false", default=None)
    p.add_argument("--no-attn", dest="ablate_attn", action="store_false", default=None)
    p.add_argument("--no-mlp", dest="ablate_mlp", action="store_false", default=None)


def _add_optimize_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--trials", type=int, default=None, help="number of optimization trials")
    p.add_argument("--n-startup", type=int, default=None, help="random trials before TPE")
    p.add_argument("--kl-weight", type=float, default=None, help="KL weight in the cost function")
    p.add_argument("--eval-harmful", type=int, default=None, help="held-out harmful prompts for refusal scoring")
    p.add_argument("--eval-harmless", type=int, default=None, help="held-out harmless prompts for KL")
    p.add_argument("--opt-max-new-tokens", type=int, default=None, help="generation length during search")
    p.add_argument("--max-weight", type=float, default=None, help="upper bound on ablation strength")
    p.add_argument("--optimize-embed", action="store_true", default=None, help="also search embedding ablation")
    p.add_argument("--opt-seed", type=int, default=None, help="optimizer seed")


def build_config(args: argparse.Namespace) -> Config:
    """Merge YAML (if any) with CLI flags; flags win when set."""
    if getattr(args, "config", None):
        cfg = Config.from_yaml(args.config)
    else:
        if not getattr(args, "model", None):
            raise SystemExit("error: --model (or --config with model.path) is required")
        cfg = Config(model=ModelConfig(path=args.model))

    m, d, a = cfg.model, cfg.data, cfg.ablation
    # Model overrides
    if getattr(args, "model", None):
        m.path = args.model
    if getattr(args, "lora", None):
        m.lora_adapter = args.lora
    if getattr(args, "tokenizer", None):
        m.tokenizer_path = args.tokenizer
    if getattr(args, "dtype", None):
        m.dtype = args.dtype
    if getattr(args, "device_map", None) is not None:
        dm = args.device_map.strip()
        m.device_map = None if dm.lower() in ("", "none") else dm
    if getattr(args, "trust_remote_code", None):
        m.trust_remote_code = True
    # Data overrides
    if getattr(args, "harmful", None):
        d.harmful = args.harmful
    if getattr(args, "harmless", None):
        d.harmless = args.harmless
    if getattr(args, "n_samples", None) is not None:
        d.n_samples = args.n_samples
    if getattr(args, "seed", None) is not None:
        d.seed = args.seed
    # Ablation overrides
    if getattr(args, "layer_index", None) is not None:
        a.layer_index = args.layer_index
    if getattr(args, "layer_fraction", None) is not None:
        a.layer_fraction = args.layer_fraction
    if getattr(args, "position", None) is not None:
        a.position = args.position
    if getattr(args, "batch_size", None) is not None:
        a.batch_size = args.batch_size
    if getattr(args, "max_length", None) is not None:
        a.max_length = args.max_length
    for flag in ("ablate_embed", "ablate_attn", "ablate_mlp"):
        val = getattr(args, flag, None)
        if val is not None:
            setattr(a, flag, val)
    # Optimize overrides
    o = cfg.optimize
    if getattr(args, "trials", None) is not None:
        o.n_trials = args.trials
    if getattr(args, "n_startup", None) is not None:
        o.n_startup_trials = args.n_startup
    if getattr(args, "kl_weight", None) is not None:
        o.kl_weight = args.kl_weight
    if getattr(args, "eval_harmful", None) is not None:
        o.n_eval_harmful = args.eval_harmful
    if getattr(args, "eval_harmless", None) is not None:
        o.n_eval_harmless = args.eval_harmless
    if getattr(args, "opt_max_new_tokens", None) is not None:
        o.max_new_tokens = args.opt_max_new_tokens
    if getattr(args, "max_weight", None) is not None:
        o.max_weight_hi = args.max_weight
    if getattr(args, "optimize_embed", None):
        o.optimize_embed = True
    if getattr(args, "opt_seed", None) is not None:
        o.seed = args.opt_seed
    # Output
    if getattr(args, "output", None):
        cfg.output_dir = args.output
    return cfg


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_directions(args) -> int:
    from .pipeline import build_directions

    cfg = build_config(args)
    _, directions = build_directions(cfg)
    out = Path(args.output or "output/directions.pt")
    directions.save(out)
    print(f"saved {directions.num_candidates} candidate directions -> {out}")
    print(f"raw norms per index: {[round(float(x), 3) for x in directions.raw_norms]}")
    return 0


def cmd_apply(args) -> int:
    from .directions import RefusalDirections
    from .model_utils import load_model_and_tokenizer
    from .pipeline import apply_and_save

    cfg = build_config(args)
    bundle = load_model_and_tokenizer(cfg.model)

    if args.directions:
        directions = RefusalDirections.load(args.directions)
    else:
        from .pipeline import build_directions
        _, directions = build_directions(cfg, bundle=bundle)

    index = directions.select_index(cfg.ablation)
    out = apply_and_save(bundle, directions.get(index), cfg, direction_index=index)
    print(f"abliterated model (direction index {index}) saved -> {out}")
    return 0


def cmd_run(args) -> int:
    from .pipeline import run

    cfg = build_config(args)
    out = run(cfg, select=args.select, evaluate=not args.no_eval)
    print(f"done -> {out}")
    return 0


def cmd_optimize(args) -> int:
    from .pipeline import optimize_and_apply

    cfg = build_config(args)
    out = optimize_and_apply(cfg)
    print(f"optimized abliterated model saved -> {out}")
    print(f"see {out}/abliteration_manifest.json for chosen params and metrics")
    return 0


def cmd_evaluate(args) -> int:
    from .directions import RefusalDirections
    from .data import load_prompts, sample_prompts
    from .evaluate import evaluate_refusal
    from .model_utils import load_model_and_tokenizer

    cfg = build_config(args)
    bundle = load_model_and_tokenizer(cfg.model)
    prompts = sample_prompts(
        load_prompts(cfg.data.harmful, default_file="harmful.txt"),
        min(cfg.data.n_samples, 32), cfg.data.seed,
    )

    direction = None
    if args.directions:
        directions = RefusalDirections.load(args.directions)
        direction = directions.get(directions.select_index(cfg.ablation))

    result = evaluate_refusal(
        bundle, prompts, direction=direction,
        model_cfg=cfg.model, max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="abliterate",
        description="Model-agnostic abliteration of refusal directions.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dir = sub.add_parser("directions", help="compute and save candidate directions")
    _add_model_args(p_dir); _add_data_args(p_dir); _add_ablation_args(p_dir)
    p_dir.add_argument("-o", "--output", help="output .pt path (default output/directions.pt)")
    p_dir.set_defaults(func=cmd_directions)

    p_apply = sub.add_parser("apply", help="orthogonalize weights and save the model")
    _add_model_args(p_apply); _add_data_args(p_apply); _add_ablation_args(p_apply)
    p_apply.add_argument("--directions", help="precomputed directions .pt (else recompute)")
    p_apply.add_argument("-o", "--output", help="output model dir (default output/)")
    p_apply.set_defaults(func=cmd_apply)

    p_run = sub.add_parser("run", help="full pipeline: directions -> select -> apply -> save")
    _add_model_args(p_run); _add_data_args(p_run); _add_ablation_args(p_run)
    p_run.add_argument("-o", "--output", help="output model dir (default output/)")
    p_run.add_argument("--select", choices=["config", "auto", "optimize"], default="config",
                       help="direction selection strategy ('optimize' = automated search)")
    p_run.add_argument("--no-eval", action="store_true", help="skip before/after refusal eval")
    _add_optimize_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_opt = sub.add_parser("optimize", help="automated abliteration: co-minimize refusals + KL")
    _add_model_args(p_opt); _add_data_args(p_opt); _add_ablation_args(p_opt)
    _add_optimize_args(p_opt)
    p_opt.add_argument("-o", "--output", help="output model dir (default output/)")
    p_opt.set_defaults(func=cmd_optimize)

    p_eval = sub.add_parser("evaluate", help="measure refusal rate (optionally ablated)")
    _add_model_args(p_eval); _add_data_args(p_eval); _add_ablation_args(p_eval)
    p_eval.add_argument("--directions", help="apply this directions file via hooks")
    p_eval.add_argument("--max-new-tokens", type=int, default=64)
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
