
import math
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
TOTAL_POINTS = 18000
REGIME_LEN = 1200
TRAIN_END = 9000
VAL_END = 12000
LOOKBACK = 64
TRAIN_WINDOWS = 7000
VAL_WINDOWS = 2000
BATCH_SIZE = 128
HIDDEN = 48
EPOCHS = 12
PATIENCE = 4
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
KAPPA = 0.90
ALPHA_MIN = 0.02
ALPHA_MAX = 1.00
NOISE_STD = 0.03
INPUT_DIM = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_series():
    rng = np.random.default_rng(2026)
    y = np.zeros(TOTAL_POINTS, dtype=np.float64)
    regimes = np.zeros(TOTAL_POINTS, dtype=np.int64)
    y[0], y[1] = 0.10, 0.12

    for t in range(2, TOTAL_POINTS):
        regime = (t // REGIME_LEN) % 3
        regimes[t] = regime
        eps = NOISE_STD * rng.standard_normal()

        if regime == 0:
            y[t] = (
                0.94*y[t-1] - 0.08*y[t-2]
                + 0.10*np.sin(0.055*t + 0.7*y[t-1]) + eps
            )
        elif regime == 1:
            y[t] = (
                0.72*y[t-1] + 0.12*y[t-2]
                + 0.22*np.sin(0.16*t + 1.2*y[t-1]) + eps
            )
        else:
            y[t] = (
                0.62*y[t-1] + 0.08*y[t-2]
                - 0.10*y[t-1]**3
                + 0.14*np.sin(0.095*t + 0.8*y[t-1]) + eps
            )
    return y.astype(np.float32), regimes

def prepare_series(y):
    mean = float(y[:TRAIN_END].mean())
    std = float(y[:TRAIN_END].std())
    ys = (y - mean) / max(std, 1e-8)
    delta = np.zeros_like(ys)
    delta[1:] = ys[1:] - ys[:-1]
    features = np.stack([ys, delta], axis=-1).astype(np.float32)
    return ys.astype(np.float32), features, mean, std

def make_windows(features, target, start, end, max_windows):
    begin = start + LOOKBACK
    finish = min(end, begin + max_windows)
    X = np.asarray(
        [features[t-LOOKBACK:t] for t in range(begin, finish)],
        dtype=np.float32
    )
    Y = np.asarray(
        [target[t] for t in range(begin, finish)],
        dtype=np.float32
    ).reshape(-1, 1)
    return X, Y

class ContractiveMatrix(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.raw = nn.Parameter(torch.empty(h, h))
        nn.init.orthogonal_(self.raw)

    def forward(self):
        fro = self.raw.norm(p="fro").clamp_min(1e-12)
        return self.raw * torch.clamp(KAPPA / fro, max=1.0)

class ExposedFuzzyController(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(2, 6),
            nn.Tanh(),
            nn.Linear(6, 1)
        )
        for m in self.score:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        init = torch.tensor([0.02, 0.10, 0.90])
        p = ((init-ALPHA_MIN)/(ALPHA_MAX-ALPHA_MIN)).clamp(1e-5, 1-1e-5)
        self.rule_logits = nn.Parameter(torch.log(p/(1-p)))
        self.register_buffer("centers", torch.tensor([-1.0, 0.0, 1.0]))
        self.log_sigma = nn.Parameter(torch.tensor(-0.5))

    def rule_values(self):
        return ALPHA_MIN + (ALPHA_MAX-ALPHA_MIN)*torch.sigmoid(self.rule_logits)

    def forward(self, x, return_internal=False):
        z = torch.tanh(self.score(x).squeeze(-1))
        d = z.unsqueeze(-1) - self.centers.view(1, 3)
        sigma = F.softplus(self.log_sigma) + 0.05
        membership = torch.exp(-d.pow(2)/(2*sigma.pow(2)))
        weights = membership / membership.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        rules = self.rule_values()
        alpha = (weights * rules.view(1,3)).sum(dim=-1)
        if return_internal:
            return alpha, z, weights, sigma, rules
        return alpha

class ExposedFACRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_h = ContractiveMatrix(HIDDEN)
        self.W_x = nn.Parameter(torch.empty(HIDDEN, INPUT_DIM))
        self.b = nn.Parameter(torch.zeros(HIDDEN))
        nn.init.xavier_uniform_(self.W_x)
        self.controller = ExposedFuzzyController()
        self.readout = nn.Sequential(
            nn.Linear(HIDDEN, 24),
            nn.Tanh(),
            nn.Linear(24, 1)
        )

    def forward(self, x, return_trace=False):
        B, T, _ = x.shape
        Wh = self.W_h()
        inp = x @ self.W_x.T + self.b
        h = torch.zeros(B, HIDDEN, device=x.device, dtype=x.dtype)
        alphas, zs, ws = [], [], []

        for t in range(T):
            xt = x[:, t, :]
            a, z, w, _, _ = self.controller(xt, return_internal=True)
            c = torch.tanh(F.linear(h, Wh) + inp[:, t, :])
            a2 = a.unsqueeze(1)
            h = (1-a2)*h + a2*c
            if return_trace:
                alphas.append(a); zs.append(z); ws.append(w)

        pred = self.readout(h)
        if return_trace:
            return pred, torch.stack(alphas, 1), torch.stack(zs, 1), torch.stack(ws, 1)
        return pred

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    s = n = 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=PIN_MEMORY)
        y = y.to(DEVICE, non_blocking=PIN_MEMORY)
        p = model(x)
        s += (p-y).pow(2).sum().item()
        n += y.numel()
    return s/max(n,1)

def train_model(model, train_loader, val_loader):
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best, best_state, best_epoch, bad = float("inf"), None, 0, 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        t0 = time.perf_counter()
        total = seen = 0

        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=PIN_MEMORY)
            y = y.to(DEVICE, non_blocking=PIN_MEMORY)
            opt.zero_grad(set_to_none=True)
            p = model(x)
            loss = F.mse_loss(p, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            total += loss.item()*x.size(0)
            seen += x.size(0)

        val = evaluate(model, val_loader)
        dt = time.perf_counter()-t0
        print(f"epoch={epoch:02d} train={total/max(seen,1):.6f} val={val:.6f} time={dt:.2f}s")

        if val < best - 1e-8:
            best, best_epoch, bad = val, epoch, 0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best, best_epoch

@torch.no_grad()
def collect_trace(model, features, target, regimes):
    model.eval()
    rows = []
    for t in range(VAL_END+LOOKBACK, TOTAL_POINTS):
        x = torch.from_numpy(features[t-LOOKBACK:t]).unsqueeze(0).to(DEVICE)
        p, aseq, zseq, wseq = model(x, return_trace=True)
        rows.append({
            "t": t,
            "regime": int(regimes[t]),
            "y": float(features[t,0]),
            "delta": float(features[t,1]),
            "abs_delta": abs(float(features[t,1])),
            "z": float(zseq[0,-1].cpu()),
            "alpha": float(aseq[0,-1].cpu()),
            "mu1": float(wseq[0,-1,0].cpu()),
            "mu2": float(wseq[0,-1,1].cpu()),
            "mu3": float(wseq[0,-1,2].cpu()),
            "prediction": float(p[0,0].cpu()),
            "target": float(target[t]),
        })
    df = pd.DataFrame(rows)
    df["abs_error"] = (df["prediction"]-df["target"]).abs()
    return df

def corr(a,b):
    a=np.asarray(a); b=np.asarray(b)
    if len(a)<3 or np.std(a)<1e-12 or np.std(b)<1e-12:
        return np.nan
    return float(np.corrcoef(a,b)[0,1])

if __name__ == "__main__":
    print("="*80)
    print("FAC-RNN COMPLETE CONTROLLER ANALYSIS")
    print("="*80)

    seed_all(SEED)
    y, regimes = generate_series()
    ys, features, mean, std = prepare_series(y)

    Xtr, Ytr = make_windows(features, ys, 0, TRAIN_END, TRAIN_WINDOWS)
    Xva, Yva = make_windows(features, ys, TRAIN_END, VAL_END, VAL_WINDOWS)

    tr = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
        batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=PIN_MEMORY, num_workers=0
    )
    va = DataLoader(
        TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva)),
        batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=PIN_MEMORY, num_workers=0
    )

    model = ExposedFACRNN().to(DEVICE)
    best_val, best_epoch = train_model(model, tr, va)

    trace = collect_trace(model, features, ys, regimes)

    with torch.no_grad():
        rules = model.controller.rule_values().cpu().numpy()
        sigma = float(F.softplus(model.controller.log_sigma).cpu() + 0.05)
        fro = float(model.W_h().norm(p="fro").cpu())

    global_summary = {
        "BestEpoch": best_epoch,
        "BestValMSE": best_val,
        "TestMSE": float((trace["prediction"]-trace["target"]).pow(2).mean()),
        "TestMAE": float(trace["abs_error"].mean()),
        "AlphaMean": float(trace["alpha"].mean()),
        "AlphaStd": float(trace["alpha"].std()),
        "ZMean": float(trace["z"].mean()),
        "ZStd": float(trace["z"].std()),
        "CorrZAbsDelta": corr(trace["z"], trace["abs_delta"]),
        "CorrAlphaAbsDelta": corr(trace["alpha"], trace["abs_delta"]),
        "CorrZAlpha": corr(trace["z"], trace["alpha"]),
        "CorrAlphaError": corr(trace["alpha"], trace["abs_error"]),
        "Rule1Alpha": float(rules[0]),
        "Rule2Alpha": float(rules[1]),
        "Rule3Alpha": float(rules[2]),
        "Sigma": sigma,
        "WFrobenius": fro,
        "Kappa": KAPPA,
        "AlphaMin": ALPHA_MIN,
        "AlphaMax": ALPHA_MAX,
        "QBound": 1.0-ALPHA_MIN*(1.0-KAPPA)
    }
    global_df = pd.DataFrame([global_summary])

    regime_rows=[]
    for r in [0,1,2]:
        s=trace[trace.regime==r]
        regime_rows.append({
            "Regime":r, "Samples":len(s),
            "AlphaMean":s.alpha.mean(), "AlphaStd":s.alpha.std(),
            "ZMean":s.z.mean(), "ZStd":s.z.std(),
            "AbsDeltaMean":s.abs_delta.mean(),
            "MAE":s.abs_error.mean(),
            "CorrZAbsDelta":corr(s.z,s.abs_delta),
            "CorrAlphaAbsDelta":corr(s.alpha,s.abs_delta)
        })
    regime_df=pd.DataFrame(regime_rows)

    transitions=[]
    for new_regime in range((VAL_END//REGIME_LEN)+1, (TOTAL_POINTS//REGIME_LEN)+1):
        trn=new_regime*REGIME_LEN
        if trn <= VAL_END or trn >= TOTAL_POINTS:
            continue
        pre=trace[(trace.t>=trn-100)&(trace.t<trn)]
        post=trace[(trace.t>=trn)&(trace.t<trn+200)]
        if len(pre)<10 or len(post)<10: continue
        transitions.append({
            "TransitionAt":trn,
            "FromRegime":new_regime-1,
            "ToRegime":new_regime,
            "PreAlpha":pre.alpha.mean(),
            "PostAlpha":post.alpha.mean(),
            "DeltaAlpha":post.alpha.mean()-pre.alpha.mean(),
            "PreZ":pre.z.mean(),
            "PostZ":post.z.mean(),
            "DeltaZ":post.z.mean()-pre.z.mean(),
            "PreMAE":pre.abs_error.mean(),
            "PostMAE":post.abs_error.mean()
        })
    trans_df=pd.DataFrame(transitions)

    # 2D policy grid
    yg=np.linspace(-2,2,101)
    dg=np.linspace(-2,2,101)
    YY,DD=np.meshgrid(yg,dg,indexing="ij")
    grid=np.stack([YY.ravel(),DD.ravel()],axis=1).astype(np.float32)

    with torch.no_grad():
        gx=torch.from_numpy(grid).to(DEVICE)
        ga,gz,gw,_,_=model.controller(gx,return_internal=True)

    grid_df=pd.DataFrame({
        "y":grid[:,0], "delta":grid[:,1],
        "z":gz.cpu().numpy(), "alpha":ga.cpu().numpy(),
        "mu1":gw[:,0].cpu().numpy(),
        "mu2":gw[:,1].cpu().numpy(),
        "mu3":gw[:,2].cpu().numpy()
    })

    global_df.to_csv("fac_complete_controller_global_summary.csv",index=False)
    regime_df.to_csv("fac_complete_controller_regime_summary.csv",index=False)
    trans_df.to_csv("fac_complete_controller_transitions.csv",index=False)
    trace.to_csv("fac_complete_controller_trace.csv",index=False)
    grid_df.to_csv("fac_complete_controller_policy_grid.csv",index=False)

    print()
    print("="*80)
    print("COMPLETE CONTROLLER ANALYSIS")
    print("="*80)
    print(global_df.to_string(index=False))

    print()
    print("REGIME SUMMARY")
    print(regime_df.to_string(index=False))

    print()
    print("TRANSITION SUMMARY")
    print(trans_df.to_string(index=False))

    print()
    print("Learned fuzzy consequents:", [float(v) for v in rules])
    print(f"Learned sigma: {sigma:.6f}")
    print(f"||W_h||F: {fro:.6f}")
    print()
    print("Saved:")
    print("  fac_complete_controller_global_summary.csv")
    print("  fac_complete_controller_regime_summary.csv")
    print("  fac_complete_controller_transitions.csv")
    print("  fac_complete_controller_trace.csv")
    print("  fac_complete_controller_policy_grid.csv")
