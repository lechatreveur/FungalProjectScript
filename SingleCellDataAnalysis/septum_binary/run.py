#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch

from torch.utils.data import DataLoader
from septum_binary.data import WindowEndpointDataset
from septum_binary.model import EndpointMILModel


# -------------------------
# Utilities
# -------------------------
def pick_device(device: str | None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return ckpt


def _bce_loss(pos_weight: float, device: str):
    # BCEWithLogitsLoss expects pos_weight as a 1D tensor
    return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))


@torch.no_grad()
def estimate_pos_rate_mc(ds: WindowEndpointDataset, endpoint: str, n_samples: int = 3000, seed: int = 0) -> float:
    """
    Estimate P(y==1) for window labels by sampling ds[i] repeatedly.
    This matches your on-the-fly window sampling and focus_p behavior.
    """
    rng = np.random.default_rng(seed)
    n = max(1, int(n_samples))
    pos = 0.0
    for _ in range(n):
        i = int(rng.integers(0, len(ds)))
        ex = ds[i]
        y = ex["y_start"].item() if endpoint == "start" else ex["y_end"].item()
        pos += float(y)
    return float(pos / n)


def bin_metrics_from_logits(logits: torch.Tensor, y: torch.Tensor, thr: float):
    """
    logits: (B,)
    y:      (B,) float {0,1}
    """
    p = (torch.sigmoid(logits) >= thr).float()
    y = y.float()

    tp = (p * y).sum().item()
    fp = (p * (1 - y)).sum().item()
    fn = ((1 - p) * y).sum().item()
    tn = ((1 - p) * (1 - y)).sum().item()

    prec = tp / max(1e-9, (tp + fp))
    rec  = tp / max(1e-9, (tp + fn))
    f1   = 2 * prec * rec / max(1e-9, (prec + rec))
    acc  = (tp + tn) / max(1e-9, (tp + tn + fp + fn))
    pred_pos_rate = (p.mean().item() if p.numel() else 0.0)
    true_pos_rate = (y.mean().item() if y.numel() else 0.0)

    return {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "pred_pos_rate": pred_pos_rate,
        "true_pos_rate": true_pos_rate,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# -------------------------
# Sliding score
# -------------------------
@torch.no_grad()
def score_strip_sliding(
    model,
    strip_uint8,
    device,
    L_max=81,
    win_len=81,
    step=1,
    invert=False,
    pool="logsumexp",
    temperature=1.0,
):
    """
    strip_uint8: (H, H*L) uint8
    returns: scores (n_windows,) sigmoid probabilities
    """
    model.eval()
    H = strip_uint8.shape[0]
    if strip_uint8.shape[1] % H != 0:
        raise ValueError(f"strip shape looks wrong: {strip_uint8.shape} (width must be multiple of H)")
    L = strip_uint8.shape[1] // H
    tiles = strip_uint8.reshape(H, L, H).transpose(1, 0, 2)  # (L,H,H)

    win_len = int(min(win_len, L_max, L))
    step = int(max(1, step))
    nwin = max(1, (L - win_len) // step + 1)

    scores = np.zeros((nwin,), dtype=np.float32)

    for k in range(nwin):
        off = k * step
        win = tiles[off:off + win_len].astype(np.float32) / 255.0
        if invert:
            win = 1.0 - win

        x = np.zeros((L_max, 1, H, H), dtype=np.float32)
        mask = np.zeros((L_max,), dtype=np.float32)
        x[:win_len, 0] = win
        mask[:win_len] = 1.0

        x_t = torch.from_numpy(x)[None].to(device)         # (1,L_max,1,H,H)
        mask_t = torch.from_numpy(mask)[None].to(device)   # (1,L_max)

        # model returns: tile_s,tile_e,win_s,win_e
        _, _, win_s, win_e = model(x_t, mask_t, pool=pool, temperature=temperature)
        # caller decides which head they are using
        # (we just return both from score_cmd)
        scores[k] = 0.0  # overwritten by score_cmd

    return scores  # placeholder (score_cmd does per-head; see below)


# -------------------------
# Train one endpoint
# -------------------------
def train_one_endpoint(args, endpoint: str):
    """
    Train one model for 'start' OR 'end' endpoint presence.
    Saves to checkpoints_binary/{endpoint}_model_latest.pt and per-epoch files.
    """
    device = pick_device(args.device)

    # Dataset: endpoint-specific
    ds_tr = WindowEndpointDataset(
        args.working_dir, "train",
        L_max=args.L_max,
        min_len=args.min_len,
        endpoint=endpoint,
        focus_p=args.focus_p,
        focus_jitter=args.focus_jitter,
        augment=True,
        invert_p=args.invert_p,
        rot90_p=args.rot90_p,
    )
    ds_va = WindowEndpointDataset(
        args.working_dir, "val",
        L_max=args.L_max,
        min_len=args.min_len,
        endpoint=endpoint,
        focus_p=0.0,          # IMPORTANT: keep val unbiased
        focus_jitter=0.0,
        augment=False,
        invert_p=0.0,
        rot90_p=0.0,
    )

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = EndpointMILModel(D=args.D).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Estimate window positive rate -> pos_weight
    pos_rate = estimate_pos_rate_mc(ds_tr, endpoint=endpoint, n_samples=args.pos_weight_mc, seed=0)
    # pos_weight = (neg/pos)
    pos_weight = float((1.0 - pos_rate) / max(1e-6, pos_rate))
    pos_weight = max(1.0, pos_weight)  # don’t go below 1
    bce = _bce_loss(pos_weight, device)

    ckpt_dir = os.path.join(args.working_dir, "training_dataset", "checkpoints_binary")
    os.makedirs(ckpt_dir, exist_ok=True)

    @torch.no_grad()
    def eval_epoch():
        model.eval()
        total = 0
        loss_sum = 0.0

        all_logits = []
        all_y = []

        for batch in dl_va:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            y = (batch["y_start"] if endpoint == "start" else batch["y_end"]).to(device)

            _, _, win_s, win_e = model(x, mask, pool=args.val_pool, temperature=1.0)
            logits = win_s if endpoint == "start" else win_e

            loss = bce(logits, y)

            b = x.size(0)
            total += b
            loss_sum += float(loss.item()) * b

            all_logits.append(logits.detach().cpu())
            all_y.append(y.detach().cpu())

        if total == 0:
            return {"loss": float("nan"), "metrics": {}}

        logits = torch.cat(all_logits, dim=0).view(-1)
        y = torch.cat(all_y, dim=0).view(-1)

        metrics = bin_metrics_from_logits(logits, y, thr=args.thr)
        return {"loss": loss_sum / total, "metrics": metrics, "n": total}

    # Train loop
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for batch in dl_tr:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            y = (batch["y_start"] if endpoint == "start" else batch["y_end"]).to(device)

            _, _, win_s, win_e = model(x, mask, pool=args.train_pool, temperature=args.temperature)
            logits = win_s if endpoint == "start" else win_e

            loss = bce(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item()) * x.size(0)
            n += x.size(0)

        tr_loss = running / max(1, n)
        va = eval_epoch()

        m = va["metrics"]
        print(
            f"[{endpoint} ep {ep:03d}] train_loss={tr_loss:.4f} val_loss={va['loss']:.4f} "
            f"acc={m.get('acc', float('nan')):.3f} prec={m.get('prec', float('nan')):.3f} "
            f"rec={m.get('rec', float('nan')):.3f} f1={m.get('f1', float('nan')):.3f} "
            f"val_pos={m.get('true_pos_rate', float('nan')):.3f} pred_pos={m.get('pred_pos_rate', float('nan')):.3f} "
            f"(pos_weight={pos_weight:.2f}, device={device})"
        )

        torch.save(
            {"state_dict": model.state_dict(), "D": args.D, "L_max": args.L_max, "endpoint": endpoint},
            os.path.join(ckpt_dir, f"{endpoint}_model_ep{ep:03d}.pt"),
        )

    torch.save(
        {"state_dict": model.state_dict(), "D": args.D, "L_max": args.L_max, "endpoint": endpoint},
        os.path.join(ckpt_dir, f"{endpoint}_model_latest.pt"),
    )
    print(f"Saved {endpoint} checkpoints to: {ckpt_dir}")


def train_cmd(args):
    # Train start and end separately (your request)
    train_one_endpoint(args, "start")
    train_one_endpoint(args, "end")


# -------------------------
# Score
# -------------------------
def score_cmd(args):
    device = pick_device(args.device)

    npz_fp = args.npz
    if not os.path.isabs(npz_fp):
        npz_fp = os.path.join(args.working_dir, npz_fp)

    with np.load(npz_fp, allow_pickle=True) as z:
        strip = np.asarray(z["strip"], dtype=np.uint8)
        true_start = int(z["start_idx"][0])
        true_end = int(z["end_idx"][0])

    ckpt_dir = os.path.join(args.working_dir, "training_dataset", "checkpoints_binary")
    ckpt_start = args.ckpt_start or os.path.join(ckpt_dir, "start_model_latest.pt")
    ckpt_end   = args.ckpt_end   or os.path.join(ckpt_dir, "end_model_latest.pt")

    model_s = EndpointMILModel(D=args.D).to(device)
    model_e = EndpointMILModel(D=args.D).to(device)
    load_checkpoint(model_s, ckpt_start, device)
    load_checkpoint(model_e, ckpt_end, device)

    def sliding_probs(model, invert: bool, which: str):
        model.eval()
        H = strip.shape[0]
        L = strip.shape[1] // H
        tiles = strip.reshape(H, L, H).transpose(1, 0, 2)

        win_len = int(min(args.win_len, args.L_max, L))
        step = int(max(1, args.step))
        nwin = max(1, (L - win_len) // step + 1)

        probs = np.zeros((nwin,), dtype=np.float32)
        for k in range(nwin):
            off = k * step
            win = tiles[off:off + win_len].astype(np.float32) / 255.0
            if invert:
                win = 1.0 - win

            x = np.zeros((args.L_max, 1, H, H), dtype=np.float32)
            mask = np.zeros((args.L_max,), dtype=np.float32)
            x[:win_len, 0] = win
            mask[:win_len] = 1.0

            x_t = torch.from_numpy(x)[None].to(device)
            mask_t = torch.from_numpy(mask)[None].to(device)

            _, _, win_s, win_e = model(x_t, mask_t, pool=args.pool, temperature=args.temperature)
            logits = win_s if which == "start" else win_e
            probs[k] = torch.sigmoid(logits)[0].item()
        return probs, win_len, nwin

    def run_one(invert: bool):
        tag = "INVERT" if invert else "NORMAL"
        p_start, win_len, nwin = sliding_probs(model_s, invert=invert, which="start")
        p_end, _, _ = sliding_probs(model_e, invert=invert, which="end")

        print(f"\n== {tag} ==")
        print(f"npz: {npz_fp}")
        print(f"true start_idx={true_start} true end_idx={true_end}")
        print(f"win_len={args.win_len} step={args.step} nwin={nwin} pool={args.pool}")

        topk = int(min(args.topk, nwin))
        s_idx = np.argsort(-p_start)[:topk]
        e_idx = np.argsort(-p_end)[:topk]

        print("Top START windows (k, prob):")
        for k in s_idx:
            print(f"  k={k:4d}  p={p_start[k]:.4f}  window=[{k*args.step},{k*args.step+win_len})")
        print("Top END windows (k, prob):")
        for k in e_idx:
            print(f"  k={k:4d}  p={p_end[k]:.4f}  window=[{k*args.step},{k*args.step+win_len})")

        if args.save_npy:
            out_prefix = os.path.splitext(os.path.basename(npz_fp))[0]
            out_dir = args.out_dir or os.path.dirname(npz_fp)
            os.makedirs(out_dir, exist_ok=True)
            pol = "invert" if invert else "normal"
            np.save(os.path.join(out_dir, f"{out_prefix}_start_{pol}.npy"), p_start)
            np.save(os.path.join(out_dir, f"{out_prefix}_end_{pol}.npy"), p_end)
            print(f"Saved npy to: {out_dir}/{out_prefix}_{{start,end}}_{pol}.npy")

    if args.both_polarities:
        run_one(False)
        run_one(True)
    else:
        run_one(args.invert)


# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser(prog="septum_binary.run")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---- train ----
    ap_tr = sub.add_parser("train", help="Train binary endpoint presence model (start/end separately).")
    ap_tr.add_argument("working_dir")
    ap_tr.add_argument("--L_max", type=int, default=81)
    ap_tr.add_argument("--min_len", type=int, default=16)
    ap_tr.add_argument("--D", type=int, default=64)
    ap_tr.add_argument("--batch_size", type=int, default=16)
    ap_tr.add_argument("--epochs", type=int, default=20)
    ap_tr.add_argument("--lr", type=float, default=1e-3)
    ap_tr.add_argument("--num_workers", type=int, default=2)
    ap_tr.add_argument("--device", type=str, default=None)

    # dataset sampling knobs (IMPORTANT)
    ap_tr.add_argument("--focus_p", type=float, default=0.75, help="Train: probability to force window to include an endpoint.")
    ap_tr.add_argument("--focus_jitter", type=float, default=0.30, help="Train: endpoint position jitter inside window (fraction of win_len).")

    # augmentation knobs
    ap_tr.add_argument("--invert_p", type=float, default=0.5)
    ap_tr.add_argument("--rot90_p", type=float, default=0.0)

    # pooling knobs
    ap_tr.add_argument("--train_pool", type=str, default="logsumexp", choices=["logsumexp", "max", "attn"])
    ap_tr.add_argument("--val_pool", type=str, default="logsumexp", choices=["logsumexp", "max", "attn"])
    ap_tr.add_argument("--temperature", type=float, default=1.0)
    ap_tr.add_argument("--thr", type=float, default=0.5)

    # pos_weight estimation
    ap_tr.add_argument("--pos_weight_mc", type=int, default=3000, help="MC samples to estimate window pos rate for pos_weight.")

    ap_tr.set_defaults(func=train_cmd)

    # ---- score ----
    ap_sc = sub.add_parser("score", help="Score one NPZ with sliding window (uses separate start/end models).")
    ap_sc.add_argument("working_dir")
    ap_sc.add_argument("--npz", required=True, help="Path to NPZ. Can be relative to working_dir.")
    ap_sc.add_argument("--ckpt_start", default=None, help="Checkpoint for start model (default: checkpoints_binary/start_model_latest.pt)")
    ap_sc.add_argument("--ckpt_end", default=None, help="Checkpoint for end model (default: checkpoints_binary/end_model_latest.pt)")
    ap_sc.add_argument("--L_max", type=int, default=81)
    ap_sc.add_argument("--D", type=int, default=64)
    ap_sc.add_argument("--win_len", type=int, default=81)
    ap_sc.add_argument("--step", type=int, default=1)
    ap_sc.add_argument("--pool", type=str, default="logsumexp", choices=["max", "logsumexp", "attn"])
    ap_sc.add_argument("--temperature", type=float, default=1.0)
    ap_sc.add_argument("--invert", action="store_true", help="Invert polarity (x -> 1-x).")
    ap_sc.add_argument("--both_polarities", action="store_true", help="Run normal + inverted and print both.")
    ap_sc.add_argument("--topk", type=int, default=10)
    ap_sc.add_argument("--save_npy", action="store_true", help="Save start/end score arrays as .npy")
    ap_sc.add_argument("--out_dir", default=None, help="Where to save npy (default: alongside NPZ)")
    ap_sc.add_argument("--device", type=str, default=None)

    ap_sc.set_defaults(func=score_cmd)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()