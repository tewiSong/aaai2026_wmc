"""Exp 3c: learned independence-assumption inference collapses on coupled structure.

The carry-chain stress test is N-digit MNIST addition under distant supervision (only the
sum is observed). A shared CNN maps each digit image to a categorical distribution; the
symbolic layer computes P(sum = label) under the independence assumption by convolving the
per-position digit distributions (the exact IA marginal of the addition constraint). As N
grows, the carry couples more digits and the IA posterior becomes multimodal, so the
training signal degrades and test accuracy collapses - the independence-assumption /
reasoning-shortcut failure that A-NeSI and NeSyDM exhibit (and which they cannot run on GO
at all). We train one model per N and report the collapse curve.

This is a real GPU training task; it loads MNIST from the local idx files.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lib

MNIST_DIR = "/ibex/user/songt/datasets/nesy2026/mnist/MNIST/raw"


# ---------------------------------------------------------------------------- MNIST loader
def _read_idx_images(path):
    with open(path, "rb") as fh:
        magic, n, rows, cols = struct.unpack(">IIII", fh.read(16))
        assert magic == 2051, f"bad magic {magic}"
        data = np.frombuffer(fh.read(), dtype=np.uint8).reshape(n, rows, cols)
    return data.astype(np.float32) / 255.0


def _read_idx_labels(path):
    with open(path, "rb") as fh:
        magic, n = struct.unpack(">II", fh.read(8))
        assert magic == 2049, f"bad magic {magic}"
        return np.frombuffer(fh.read(), dtype=np.uint8).astype(np.int64)


def load_mnist(train=True):
    pre = "train" if train else "t10k"
    x = _read_idx_images(os.path.join(MNIST_DIR, f"{pre}-images-idx3-ubyte"))
    y = _read_idx_labels(os.path.join(MNIST_DIR, f"{pre}-labels-idx1-ubyte"))
    return x, y


# ------------------------------------------------------------------------------- the model
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # logits over 10 digits


def number_distribution(digit_probs):
    """Distribution over an N-digit number's integer value under the IA.

    digit_probs: (B, N, 10), most significant digit first.
    Returns (B, 10**N) distribution over the integer value by independent-digit product.
    """
    B, N, _ = digit_probs.shape
    dist = digit_probs[:, 0, :]  # (B,10) value 0..9
    for i in range(1, N):
        # new value = old*10 + digit ; outer product then reshape
        nd = digit_probs[:, i, :]  # (B,10)
        dist = (dist.unsqueeze(2) * nd.unsqueeze(1)).reshape(B, -1)
    return dist  # (B, 10**N)


def sum_marginal(distA, distB, max_sum):
    """P(a+b = s) by convolving two integer distributions (the addition constraint).

    Linear convolution via real FFT: P(a+b=.) = irfft(rfft(distA) * rfft(distB)).
    """
    na = distA.shape[1]
    nb = distB.shape[1]
    L = na + nb - 1  # == max_sum + 1
    fa = torch.fft.rfft(distA, n=L)
    fb = torch.fft.rfft(distB, n=L)
    conv = torch.fft.irfft(fa * fb, n=L)
    return conv.clamp_min(0.0)


def sample_marginal_loss(probs, N, target, n_samples):
    """REINFORCE estimate of -log P(sum = target) from `n_samples` digit samples.

    This is the naive Monte-Carlo WMC estimate that learned approximators (A-NeSI,
    NeSyDM) are built to replace: sample digit assignments from the per-digit
    distributions, score the ones whose sum matches the target, and take the
    score-function gradient. As N grows the probability that any sample matches the
    target sum vanishes, so almost every batch yields no matching sample and no gradient
    - the independence-assumption collapse on coupled structure.

    probs: (B, 2N, 10); target: (B,) integer sums.
    """
    B = probs.shape[0]
    dist = torch.distributions.Categorical(probs=probs)  # (B,2N)
    logp_all = 0.0
    reward_logp = torch.zeros(B, device=probs.device)
    hit_count = torch.zeros(B, device=probs.device)
    for _ in range(n_samples):
        s = dist.sample()  # (B,2N)
        logp = dist.log_prob(s).sum(dim=1)  # (B,)
        a = torch.zeros(B, dtype=torch.long, device=probs.device)
        b = torch.zeros(B, dtype=torch.long, device=probs.device)
        for i in range(N):
            a = a * 10 + s[:, i]
            b = b * 10 + s[:, N + i]
        hit = ((a + b) == target).float()
        reward_logp = reward_logp + hit * logp
        hit_count = hit_count + hit
    # maximize E[1{correct} log p]  ==> minimize -reward/n_samples
    loss = -(reward_logp / n_samples).mean()
    frac_hit = (hit_count > 0).float().mean().item()
    return loss, frac_hit


def run_for_N(N, device, epochs=3, batch=128, n_train=20000, n_test=4000, seed=0,
              mode="exact", n_samples=64):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xtr, ytr = load_mnist(train=True)
    xte, yte = load_mnist(train=False)

    def make_pairs(x, y, n_examples):
        # each example: two N-digit numbers -> 2N images; label = integer sum
        idx = rng.integers(0, len(x), size=(n_examples, 2 * N))
        imgs = x[idx]  # (n,2N,28,28)
        digs = y[idx]  # (n,2N)
        a = np.zeros(n_examples, dtype=np.int64)
        b = np.zeros(n_examples, dtype=np.int64)
        for i in range(N):
            a = a * 10 + digs[:, i]
            b = b * 10 + digs[:, N + i]
        return imgs, digs, (a + b)

    tr_imgs, tr_digs, tr_sum = make_pairs(xtr, ytr, n_train)
    te_imgs, te_digs, te_sum = make_pairs(xte, yte, n_test)
    max_sum = 2 * (10 ** N - 1)

    model = DigitCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    tr_imgs_t = torch.tensor(tr_imgs).to(device)
    tr_sum_t = torch.tensor(tr_sum).to(device)

    n = tr_imgs.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        model.train()
        tot = 0.0
        for s in range(0, n, batch):
            bi = perm[s:s + batch]
            imgs = tr_imgs_t[bi]  # (b,2N,28,28)
            bsz = imgs.shape[0]
            logits = model(imgs.reshape(bsz * 2 * N, 1, 28, 28))
            probs = F.softmax(logits, dim=1).reshape(bsz, 2 * N, 10)
            target = tr_sum_t[bi]
            if mode == "exact":
                distA = number_distribution(probs[:, :N, :])
                distB = number_distribution(probs[:, N:, :])
                psum = sum_marginal(distA, distB, max_sum)  # (b, max_sum+1)
                p_correct = psum.gather(1, target.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
                loss = -torch.log(p_correct).mean()
                frac_hit = 1.0
            elif mode == "sample":
                loss, frac_hit = sample_marginal_loss(probs, N, target, n_samples)
            else:
                raise ValueError(f"unknown mode {mode}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * bsz
        print(f"  [N={N},{mode}] epoch {ep+1}/{epochs} loss={tot/n:.4f} frac_hit={frac_hit:.3f}",
              flush=True)

    # evaluation: predict each digit by argmax, compute sum, exact-match accuracy
    model.eval()
    te_imgs_t = torch.tensor(te_imgs).to(device)
    with torch.no_grad():
        bsz = te_imgs.shape[0]
        logits = model(te_imgs_t.reshape(bsz * 2 * N, 1, 28, 28))
        pred = logits.argmax(1).reshape(bsz, 2 * N).cpu().numpy()
    a = np.zeros(bsz, dtype=np.int64)
    b = np.zeros(bsz, dtype=np.int64)
    for i in range(N):
        a = a * 10 + pred[:, i]
        b = b * 10 + pred[:, N + i]
    sum_acc = float(np.mean((a + b) == te_sum))
    digit_acc = float(np.mean(pred == te_digs))
    print(f"[N={N},{mode}] test sum-accuracy={sum_acc:.3f} digit-accuracy={digit_acc:.3f}",
          flush=True)
    return dict(N=N, mode=mode, sum_accuracy=sum_acc, digit_accuracy=digit_acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", default="1,2,3,4")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--modes", default="exact,sample")
    ap.add_argument("--n_samples", type=int, default=64)
    args = ap.parse_args()
    lib.ensure_results_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    results = []
    for mode in args.modes.split(","):
        for N in [int(x) for x in args.Ns.split(",")]:
            for seed in [int(s) for s in args.seeds.split(",")]:
                t0 = time.time()
                r = run_for_N(N, device, epochs=args.epochs, seed=seed,
                              mode=mode, n_samples=args.n_samples)
                r["seed"] = seed
                r["time_s"] = round(time.time() - t0, 1)
                results.append(r)

    # aggregate mean/std over seeds, per mode
    agg = {}
    for mode in sorted(set(r["mode"] for r in results)):
        agg[mode] = {}
        for N in sorted(set(r["N"] for r in results if r["mode"] == mode)):
            accs = [r["sum_accuracy"] for r in results if r["N"] == N and r["mode"] == mode]
            agg[mode][N] = dict(mean=float(np.mean(accs)), std=float(np.std(accs)),
                                min=float(np.min(accs)), max=float(np.max(accs)))
            print(f"[carry-chain] {mode} N={N}: sum-acc "
                  f"{agg[mode][N]['mean']:.3f} +/- {agg[mode][N]['std']:.3f}", flush=True)
    out = os.path.join(lib.RESULTS_DIR, "exp3c_carrychain.json")
    with open(out, "w") as fh:
        json.dump(dict(per_run=results, aggregate=agg), fh, indent=2)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
