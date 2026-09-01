# FAC-RNN Stability Ablation V1
# A Fixed alpha + contraction
# B Adaptive alpha + contraction
# C Fuzzy adaptive alpha + contraction
# D Fuzzy adaptive alpha without contraction

import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SEEDS = [42, 123, 456]
LOOKBACK = 64
HIDDEN = 48
N_TRAIN, N_VAL, N_TEST = 9000, 2500, 2500
EPOCHS = 20
BATCH_SIZE = 128
LR = 2e-3
WEIGHT_DECAY = 1e-5

KAPPA = 0.9
ALPHA_MIN = 0.02
ALPHA_MAX = 0.98
FUZZY_SIGMA = 0.55

OUT_DIR = Path("fac_stability_ablation_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_series(n, seed):
    """Self-contained nonlinear switching series.
    For exact replication, replace this body with the generator from the
    established forecasting experiment used in the previous FAC-RNN runs.
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.float32)
    for t in range(2, n):
        r = (t // 1200) % 3
        if r == 0:
            a1, a2, nl, forcing = .72, -.18, .10, .16
        elif r == 1:
            a1, a2, nl, forcing = .42, .34, .18, .11
        else:
            a1, a2, nl, forcing = .80, -.32, .07, .20
        seasonal = .08 * math.sin(2 * math.pi * t / 80.0)
        x = forcing * math.sin(2 * math.pi * t / 37.0) + seasonal
        y[t] = (a1 * y[t-1] + a2 * y[t-2]
                + nl * math.tanh(y[t-1]) + x
                + .03 * rng.normal())
    y = (y - y.mean()) / (y.std() + 1e-8)
    return y.astype(np.float32)


def make_windows(series, lookback):
    X, Y = [], []
    for t in range(lookback, len(series)):
        X.append(series[t-lookback:t])
        Y.append(series[t])
    return np.asarray(X, np.float32)[..., None], np.asarray(Y, np.float32)[..., None]


class BaseRNN(nn.Module):
    def __init__(self, hidden=48, contraction=True):
        super().__init__()
        self.hidden = hidden
        self.use_contraction = contraction
        self.W_h_raw = nn.Parameter(torch.empty(hidden, hidden))
        self.W_x = nn.Parameter(torch.empty(hidden, 1))
        self.b = nn.Parameter(torch.zeros(hidden))
        self.readout = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.W_h_raw)
        nn.init.xavier_uniform_(self.W_x)

    def W_h(self):
        if not self.use_contraction:
            return self.W_h_raw
        n = torch.linalg.matrix_norm(self.W_h_raw, ord="fro")
        scale = torch.clamp(KAPPA / (n + 1e-12), max=1.0)
        return self.W_h_raw * scale

    def fro_norm(self):
        with torch.no_grad():
            return float(torch.linalg.matrix_norm(self.W_h(), ord="fro"))

    def spectral_norm(self):
        with torch.no_grad():
            return float(torch.linalg.matrix_norm(self.W_h(), ord=2))


class FixedAlphaRNN(BaseRNN):
    def __init__(self, hidden=48):
        super().__init__(hidden, True)
        self.alpha_value = 0.40

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden, device=x.device)
        wh = self.W_h()
        aa = []
        for t in range(T):
            xt = x[:, t, :]
            proposal = torch.tanh(h @ wh.T + xt @ self.W_x.T + self.b)
            a = torch.full((B, 1), self.alpha_value, device=x.device)
            h = (1-a) * h + a * proposal
            aa.append(a)
        return self.readout(h), torch.cat(aa, 1)


class AdaptiveRNN(BaseRNN):
    def __init__(self, hidden=48):
        super().__init__(hidden, True)
        self.net = nn.Sequential(nn.Linear(2, 24), nn.Tanh(), nn.Linear(24, 1))

    def alpha(self, xt, h):
        z = torch.cat([xt, h.mean(1, keepdim=True)], 1)
        return ALPHA_MIN + (ALPHA_MAX-ALPHA_MIN) * torch.sigmoid(self.net(z))

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden, device=x.device)
        wh = self.W_h()
        aa = []
        for t in range(T):
            xt = x[:, t, :]
            proposal = torch.tanh(h @ wh.T + xt @ self.W_x.T + self.b)
            a = self.alpha(xt, h)
            h = (1-a) * h + a * proposal
            aa.append(a)
        return self.readout(h), torch.cat(aa, 1)


class FuzzyAdaptiveRNN(BaseRNN):
    def __init__(self, hidden=48, contraction=True):
        super().__init__(hidden, contraction)
        self.score = nn.Sequential(nn.Linear(2, 24), nn.Tanh(), nn.Linear(24, 1))
        self.register_buffer("centers", torch.tensor([-1., 0., 1.]))
        self.rule_alpha = nn.Parameter(torch.tensor([ALPHA_MIN, .10, .92]))

    def alpha_details(self, xt, h):
        z = self.score(torch.cat([xt, h.mean(1, keepdim=True)], 1))
        d2 = (z.unsqueeze(-1) - self.centers.view(1, 1, -1)) ** 2
        mu = torch.exp(-0.5 * d2 / (FUZZY_SIGMA**2))
        mu = mu / (mu.sum(-1, keepdim=True) + 1e-8)
        a_rules = self.rule_alpha.clamp(ALPHA_MIN, ALPHA_MAX)
        a = (mu * a_rules.view(1, 1, -1)).sum(-1)
        return a, z, mu

    def forward(self, x, details=False):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden, device=x.device)
        wh = self.W_h()
        aa, zz, mm = [], [], []
        for t in range(T):
            xt = x[:, t, :]
            proposal = torch.tanh(h @ wh.T + xt @ self.W_x.T + self.b)
            a, z, mu = self.alpha_details(xt, h)
            h = (1-a) * h + a * proposal
            aa.append(a); zz.append(z); mm.append(mu)
        y = self.readout(h)
        if details:
            return y, torch.cat(aa,1), torch.cat(zz,1), torch.cat(mm,1)
        return y, torch.cat(aa,1)


def batches(X, Y, seed, epoch):
    idx = np.arange(len(X))
    np.random.default_rng(seed + 1009*epoch).shuffle(idx)
    for s in range(0, len(idx), BATCH_SIZE):
        j = idx[s:s+BATCH_SIZE]
        yield torch.from_numpy(X[j]).to(DEVICE), torch.from_numpy(Y[j]).to(DEVICE)


@torch.no_grad()
def evaluate(model, X, Y):
    model.eval()
    pred = []
    for s in range(0, len(X), BATCH_SIZE):
        xb = torch.from_numpy(X[s:s+BATCH_SIZE]).to(DEVICE)
        pred.append(model(xb)[0].cpu().numpy())
    pred = np.concatenate(pred)
    return float(np.mean((pred-Y)**2)), float(np.mean(np.abs(pred-Y)))


def fit(model, Xtr, Ytr, Xv, Yv, seed, name):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best = float("inf")
    state = None
    times = []
    for ep in range(1, EPOCHS+1):
        model.train()
        t0 = time.perf_counter()
        for xb, yb in batches(Xtr, Ytr, seed, ep):
            opt.zero_grad(set_to_none=True)
            pred, _ = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        times.append(time.perf_counter()-t0)
        vmse, _ = evaluate(model, Xv, Yv)
        if vmse < best:
            best = vmse
            state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"[{name}] seed={seed} epoch={ep:02d} val_mse={vmse:.8f} time={times[-1]:.2f}s")
    model.load_state_dict(state)
    return best, float(np.mean(times))


@torch.no_grad()
def perturb_gain(model, X, eps=1e-4):
    model.eval()
    gains = []
    for s in range(0, min(len(X), 1024), BATCH_SIZE):
        xb = torch.from_numpy(X[s:s+BATCH_SIZE]).to(DEVICE)
        xp = xb.clone()
        xp[:,0,0] += eps
        y1 = model(xb)[0]
        y2 = model(xp)[0]
        gains.extend((torch.abs(y2-y1)/eps).cpu().numpy().ravel())
    g = np.asarray(gains)
    return float(g.mean()), float(np.median(g)), float(np.percentile(g,95)), float(g.max())


def fuzzy_stats(model, X):
    if not isinstance(model, FuzzyAdaptiveRNN):
        return {}
    al, zz, mu = [], [], []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(X), BATCH_SIZE):
            xb = torch.from_numpy(X[s:s+BATCH_SIZE]).to(DEVICE)
            _, a, z, m = model(xb, True)
            al.append(a.cpu().numpy().ravel())
            zz.append(z.cpu().numpy().ravel())
            mu.append(m.cpu().numpy().reshape(-1,3))
    a = np.concatenate(al); z = np.concatenate(zz); m = np.concatenate(mu)
    return dict(alpha_mean=float(a.mean()), alpha_std=float(a.std()),
                alpha_min=float(a.min()), alpha_max=float(a.max()),
                z_mean=float(z.mean()), z_std=float(z.std()),
                rule1_share=float(m[:,0].mean()),
                rule2_share=float(m[:,1].mean()),
                rule3_share=float(m[:,2].mean()))


def main():
    print("="*78)
    print("FAC-RNN STABILITY ABLATION V1")
    print("Device:", DEVICE)
    print("="*78)

    total = LOOKBACK + N_TRAIN + N_VAL + N_TEST
    rows, fs_rows = [], []

    for seed in SEEDS:
        seed_all(seed)
        series = make_series(total, seed)
        X, Y = make_windows(series, LOOKBACK)

        Xtr, Ytr = X[:N_TRAIN], Y[:N_TRAIN]
        Xv, Yv = X[N_TRAIN:N_TRAIN+N_VAL], Y[N_TRAIN:N_TRAIN+N_VAL]
        s0 = N_TRAIN + N_VAL
        Xt, Yt = X[s0:s0+N_TEST], Y[s0:s0+N_TEST]

        configs = [
            ("A_fixed_contractive", lambda: FixedAlphaRNN(HIDDEN)),
            ("B_adaptive_contractive", lambda: AdaptiveRNN(HIDDEN)),
            ("C_fuzzy_contractive", lambda: FuzzyAdaptiveRNN(HIDDEN, True)),
            ("D_fuzzy_unconstrained", lambda: FuzzyAdaptiveRNN(HIDDEN, False)),
        ]

        for name, make_model in configs:
            seed_all(seed)
            model = make_model().to(DEVICE)
            best_val, sec = fit(model, Xtr, Ytr, Xv, Yv, seed, name)
            tmse, tmae = evaluate(model, Xt, Yt)
            pg = perturb_gain(model, Xt)
            row = dict(model=name, seed=seed, best_val_mse=best_val,
                       test_mse=tmse, test_mae=tmae, mean_epoch_sec=sec,
                       fro_norm=model.fro_norm(), spectral_norm=model.spectral_norm(),
                       perturb_gain_mean=pg[0], perturb_gain_median=pg[1],
                       perturb_gain_p95=pg[2], perturb_gain_max=pg[3])
            rows.append(row)
            fs = fuzzy_stats(model, Xt)
            if fs:
                fs_rows.append(dict(model=name, seed=seed, **fs))
            print(f"RESULT {name} seed={seed}: MSE={tmse:.8f} MAE={tmae:.8f} "
                  f"Fro={row['fro_norm']:.6f} Spec={row['spectral_norm']:.6f}")

    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR/"fac_stability_ablation_results.csv", index=False)

    if fs_rows:
        pd.DataFrame(fs_rows).to_csv(OUT_DIR/"fac_stability_controller_stats.csv", index=False)

    summary = (results.groupby("model")
               .agg(test_mse_mean=("test_mse","mean"), test_mse_std=("test_mse","std"),
                    test_mae_mean=("test_mae","mean"), test_mae_std=("test_mae","std"),
                    fro_mean=("fro_norm","mean"), fro_max=("fro_norm","max"),
                    spectral_mean=("spectral_norm","mean"), spectral_max=("spectral_norm","max"),
                    perturb_gain_mean=("perturb_gain_mean","mean"),
                    perturb_gain_p95=("perturb_gain_p95","mean"),
                    perturb_gain_max=("perturb_gain_max","max"))
               .reset_index())
    summary.to_csv(OUT_DIR/"fac_stability_ablation_summary.csv", index=False)

    means = summary.set_index("model")["test_mse_mean"]
    pairs = [
        ("A_fixed_contractive","B_adaptive_contractive"),
        ("B_adaptive_contractive","C_fuzzy_contractive"),
        ("C_fuzzy_contractive","D_fuzzy_unconstrained"),
    ]
    comp = []
    for base, new in pairs:
        b, n = means[base], means[new]
        comp.append(dict(baseline=base, new_model=new,
                         baseline_mse=b, new_mse=n,
                         relative_mse_reduction_percent=100*(b-n)/b))
    pd.DataFrame(comp).to_csv(OUT_DIR/"fac_stability_pairwise_comparisons.csv", index=False)

    print("\nFINAL SUMMARY")
    print(summary.to_string(index=False))
    print("\nPAIRWISE")
    print(pd.DataFrame(comp).to_string(index=False))
    print("\nSaved:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
