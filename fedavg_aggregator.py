import argparse, json, math
from pathlib import Path
from typing import List, Dict, Optional
import torch
import torch.nn as nn
from torchvision import models

def build_model(num_labels: int = 14, weights: str = "IMAGENET1K_V1", device: str = "cpu") -> nn.Module:
    m = models.densenet121(weights=weights)
    m.classifier = nn.Sequential(
        nn.Linear(m.classifier.in_features, num_labels),
        nn.Sigmoid()
    )
    return m.to(device)

def load_state_dict(path: Path, map_location="cpu") -> Dict[str, torch.Tensor]:
    sd = torch.load(path, map_location=map_location)
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        return sd["state_dict"]
    if isinstance(sd, dict):
        return sd
    raise ValueError(f"Unsupported checkpoint format: {path}")

def check_compat(keys_list: List[List[str]]):
    ref = keys_list[0]
    for i, ks in enumerate(keys_list[1:], start=2):
        if ref != ks:
            missing = set(ref) - set(ks)
            extra   = set(ks) - set(ref)
            msg = []
            if missing: msg.append(f"missing in model{i}: {sorted(list(missing))[:5]} ...")
            if extra:   msg.append(f"extra in model{i}: {sorted(list(extra))[:5]} ...")
            raise RuntimeError("State dict key mismatch: " + " | ".join(msg))

def fedavg(state_dicts: List[Dict[str, torch.Tensor]], weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
    klist = [list(sd.keys()) for sd in state_dicts]
    check_compat(klist)
    if weights is None:
        weights = [1.0] * len(state_dicts)
    w_sum = float(sum(weights))
    if w_sum <= 0 or any(math.isnan(w) or math.isinf(w) for w in weights):
        raise ValueError("Invalid FedAvg weights.")
    out = {}
    keys = klist[0]
    for k in keys:
        acc = None
        for sd, w in zip(state_dicts, weights):
            t = sd[k]
            t = t.float() if t.is_floating_point() else t
            acc = (t * (w / w_sum)) if acc is None else (acc + t * (w / w_sum))
        out[k] = acc
    return out

def maybe_cast_dtype(sd: Dict[str, torch.Tensor], dtype: Optional[str]) -> Dict[str, torch.Tensor]:
    if not dtype: return sd
    target = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(dtype.lower())
    if target is None: return sd
    casted = {}
    for k, v in sd.items():
        casted[k] = v.to(target) if v.is_floating_point() else v
    return casted

def parse_weights_arg(weights_arg: Optional[str], n: int) -> Optional[List[float]]:
    if not weights_arg:
        return None
    vals = [float(x.strip()) for x in weights_arg.split(",")]
    if len(vals) != n:
        raise ValueError(f"--weights has {len(vals)} entries, but {n} checkpoints provided.")
    return vals

def load_sample_counts_from_manifests(manifest_paths: List[Path]) -> Optional[List[float]]:
    if not manifest_paths: return None
    counts = []
    for p in manifest_paths:
        if not p.exists(): return None
        with open(p) as f:
            j = json.load(f)
        # allow keys: num_samples or samples
        cnt = j.get("num_samples", j.get("samples"))
        if cnt is None: return None
        counts.append(float(cnt))
    return counts

def main():
    ap = argparse.ArgumentParser(description="FedAvg aggregator for DenseNet121 CheXpert (torchvision).")
    ap.add_argument("--ckpts", nargs="+", required=True, help="Client checkpoint paths (state_dict .pt files).")
    ap.add_argument("--out", required=True, help="Output global checkpoint path.")
    ap.add_argument("--dtype", choices=["fp32","fp16","bf16"], default="fp32", help="Output floating dtype for tensors.")
    ap.add_argument("--weights", help="Comma-separated sample counts to weight FedAvg (e.g. '1024,980,765,1200').")
    ap.add_argument("--manifests", nargs="*", help="Optional JSON manifests per client with {'num_samples': int}.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save-strict", action="store_true", help="Save full model (architecture+weights) instead of state_dict.")
    args = ap.parse_args()

    ckpt_paths = [Path(p) for p in args.ckpts]
    for p in ckpt_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing checkpoint: {p}")

    # Build skeleton model to (optionally) save full model later or to sanity-load.
    model = build_model(device=args.device)

    # Load state dicts
    sds = [load_state_dict(p, map_location="cpu") for p in ckpt_paths]

    # Determine FedAvg weights
    w = parse_weights_arg(args.weights, len(sds)) if args.weights else None
    if w is None and args.manifests:
        w = load_sample_counts_from_manifests([Path(m) for m in args.manifests])
    # fallback: equal weights
    if w is None:
        w = [1.0] * len(sds)

    # Aggregate
    global_sd = fedavg(sds, w)
    global_sd = maybe_cast_dtype(global_sd, args.dtype)

    # (Optional) quick key check by loading into model
    missing, unexpected = model.load_state_dict(global_sd, strict=False)
    if missing or unexpected:
        print(f"[WARN] missing keys: {missing[:5]} ... | unexpected: {unexpected[:5]} ...")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.save_strict:
        torch.save(model, out_path)
    else:
        torch.save(global_sd, out_path)

    # Emit a small summary
    summary = {
        "num_clients": len(sds),
        "weights": w,
        "dtype": args.dtype,
        "out": str(out_path)
    }
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()