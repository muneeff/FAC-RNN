
# ============================================================
# FAC-RNN REGIME-SHIFT v1
#
# Goal:
#   Test whether fuzzy adaptive time-scale responds to an abrupt
#   change in local dynamics better than:
#       A) fixed alpha
#       B) learned non-fuzzy alpha
#
# Core recurrence (unchanged):
#
#   h_t = (1-alpha_t) h_(t-1)
#         + alpha_t tanh(W_h h_(t-1) + W_x x_t + b)
#
# Stability:
#
#   ||W_h||_2 <= ||W_h||_F <= KAPPA = 0.90
#   alpha_t >= ALPHA_MIN = 0.02
#
# Therefore:
#
#   q <= 1 - alpha_min*(1-kappa) = 0.998
#
# Experiment:
#   A single forecasting stream changes abruptly from a slow regime
#   to a fast regime and then to a nonlinear regime.
#
#   We train on the first part and evaluate on a held-out test
#   stream containing known regime transitions.
#
# Metrics:
#   - one-step MSE/MAE
#   - alpha mean by regime
#   - alpha change around regime transitions
#   - adaptation lag
#   - pre/post transition error
#
# IMPORTANT:
#   The controller does NOT receive the true regime label.
#   It sees only [y_t, delta_t].
#
# This is a mechanistic test, not a final benchmark.
# ============================================================

import time
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Configuration
# ============================================================

SEED = 42

TOTAL_POINTS = 18000

# Deliberate abrupt regime boundaries.
REGIME_LEN = 1200

LOOKBACK = 64

TRAIN_END = 9000
VAL_END = 12000

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


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PIN_MEMORY = DEVICE.type == "cuda"

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_all(SEED)


# ============================================================
# Regime-shift series
# ============================================================

def generate_series():

    rng = np.random.default_rng(
        2026
    )

    y = np.zeros(
        TOTAL_POINTS,
        dtype=np.float64
    )

    regimes = np.zeros(
        TOTAL_POINTS,
        dtype=np.int64
    )

    y[0] = 0.10
    y[1] = 0.12

    for t in range(
        2,
        TOTAL_POINTS
    ):

        regime = (
            t // REGIME_LEN
        ) % 3

        regimes[t] = regime

        eps = (
            NOISE_STD
            *
            rng.standard_normal()
        )

        if regime == 0:
            # Slow, persistent regime.
            y[t] = (
                0.94 * y[t-1]
                -
                0.08 * y[t-2]
                +
                0.10 * np.sin(
                    0.055*t
                    +
                    0.7*y[t-1]
                )
                +
                eps
            )

        elif regime == 1:
            # Fast regime.
            y[t] = (
                0.72 * y[t-1]
                +
                0.12 * y[t-2]
                +
                0.22 * np.sin(
                    0.16*t
                    +
                    1.2*y[t-1]
                )
                +
                eps
            )

        else:
            # Nonlinear regime.
            y[t] = (
                0.62 * y[t-1]
                +
                0.08 * y[t-2]
                -
                0.10 * y[t-1]**3
                +
                0.14 * np.sin(
                    0.095*t
                    +
                    0.8*y[t-1]
                )
                +
                eps
            )

    return (
        y.astype(np.float32),
        regimes
    )


# ============================================================
# Features and windows
# ============================================================

def prepare_series():

    y, regimes = generate_series()

    train_y = y[:TRAIN_END]

    mean = float(
        train_y.mean()
    )

    std = float(
        train_y.std()
    )

    ys = (
        y - mean
    ) / max(
        std,
        1e-8
    )

    delta = np.zeros_like(
        ys
    )

    delta[1:] = (
        ys[1:]
        -
        ys[:-1]
    )

    features = np.stack(
        [
            ys,
            delta
        ],
        axis=-1
    ).astype(
        np.float32
    )

    return (
        y,
        ys.astype(np.float32),
        features,
        regimes,
        mean,
        std
    )


def build_windows(
    features,
    target,
    start,
    end,
    max_windows=None
):

    begin = (
        start + LOOKBACK
    )

    finish = end

    if max_windows is not None:
        finish = min(
            finish,
            begin + max_windows
        )

    X = []
    Y = []

    for t in range(
        begin,
        finish
    ):

        X.append(
            features[
                t-LOOKBACK:t
            ]
        )

        Y.append(
            target[t]
        )

    return (
        np.asarray(
            X,
            dtype=np.float32
        ),
        np.asarray(
            Y,
            dtype=np.float32
        ).reshape(
            -1,1
        )
    )


# ============================================================
# Models
# ============================================================

class ContractiveMatrix(nn.Module):

    def __init__(self, h):
        super().__init__()

        self.raw = nn.Parameter(
            torch.empty(h,h)
        )

        nn.init.orthogonal_(
            self.raw
        )

    def forward(self):

        f = self.raw.norm(
            p='fro'
        ).clamp_min(
            1e-12
        )

        return (
            self.raw
            *
            torch.clamp(
                KAPPA / f,
                max=1.0
            )
        )


class FixedController(nn.Module):

    def __init__(
        self,
        alpha=0.10
    ):
        super().__init__()

        self.register_buffer(
            'alpha',
            torch.tensor(
                float(alpha)
            )
        )

    def forward(self, x):

        return self.alpha.expand(
            x.shape[:-1]
        )


class NonFuzzyController(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                INPUT_DIM,
                8
            ),
            nn.Tanh(),
            nn.Linear(
                8,
                1
            )
        )

        for m in self.net:

            if isinstance(
                m,
                nn.Linear
            ):

                nn.init.xavier_uniform_(
                    m.weight
                )

                nn.init.zeros_(
                    m.bias
                )

    def forward(self, x):

        z = self.net(
            x
        ).squeeze(-1)

        return (
            ALPHA_MIN
            +
            (
                ALPHA_MAX
                -
                ALPHA_MIN
            )
            *
            torch.sigmoid(z)
        )


class FuzzyController(nn.Module):

    def __init__(self):

        super().__init__()

        # Compact fuzzy context encoder.
        self.score = nn.Sequential(
            nn.Linear(
                INPUT_DIM,
                6
            ),
            nn.Tanh(),
            nn.Linear(
                6,
                1
            )
        )

        for m in self.score:

            if isinstance(
                m,
                nn.Linear
            ):

                nn.init.xavier_uniform_(
                    m.weight
                )

                nn.init.zeros_(
                    m.bias
                )

        init = torch.tensor(
            [
                0.02,
                0.10,
                0.90
            ]
        )

        p = (
            (
                init
                -
                ALPHA_MIN
            )
            /
            (
                ALPHA_MAX
                -
                ALPHA_MIN
            )
        ).clamp(
            1e-5,
            1.0-1e-5
        )

        self.rule_logits = nn.Parameter(
            torch.log(
                p/(1-p)
            )
        )

        self.register_buffer(
            'centers',
            torch.tensor(
                [-1.0,0.0,1.0]
            )
        )

        self.log_sigma = nn.Parameter(
            torch.tensor(-0.5)
        )

    def rule_values(self):

        return (
            ALPHA_MIN
            +
            (
                ALPHA_MAX
                -
                ALPHA_MIN
            )
            *
            torch.sigmoid(
                self.rule_logits
            )
        )

    def forward(self, x):

        z = torch.tanh(
            self.score(x).squeeze(-1)
        )

        d = (
            z.unsqueeze(-1)
            -
            self.centers.view(1,3)
        )

        sigma = (
            F.softplus(
                self.log_sigma
            )
            +
            0.05
        )

        membership = torch.exp(
            -d.pow(2)
            /
            (
                2*sigma.pow(2)
            )
        )

        weights = membership / (
            membership.sum(
                dim=-1,
                keepdim=True
            ).clamp_min(1e-8)
        )

        rules = self.rule_values()

        alpha = (
            weights
            *
            rules.view(1,3)
        ).sum(
            dim=-1
        )

        return alpha


class ForecastRNN(nn.Module):

    def __init__(
        self,
        controller
    ):

        super().__init__()

        self.controller_type = controller

        self.W_h = ContractiveMatrix(
            HIDDEN
        )

        self.W_x = nn.Parameter(
            torch.empty(
                HIDDEN,
                INPUT_DIM
            )
        )

        self.b = nn.Parameter(
            torch.zeros(HIDDEN)
        )

        nn.init.xavier_uniform_(
            self.W_x
        )

        if controller == "fixed":
            self.ctrl = FixedController()

        elif controller == "nonfuzzy":
            self.ctrl = NonFuzzyController()

        elif controller == "fuzzy":
            self.ctrl = FuzzyController()

        else:
            raise ValueError(controller)

        self.readout = nn.Sequential(
            nn.Linear(
                HIDDEN,
                24
            ),
            nn.Tanh(),
            nn.Linear(
                24,
                1
            )
        )

    def forward(
        self,
        x,
        return_alpha=False
    ):

        B, T, _ = x.shape

        wh = self.W_h()

        inp = (
            x
            @
            self.W_x.T
            +
            self.b
        )

        h = torch.zeros(
            B,
            HIDDEN,
            device=x.device,
            dtype=x.dtype
        )

        alpha_seq = []

        for t in range(T):

            xt = x[:,t,:]

            alpha = self.ctrl(
                xt
            )

            alpha_seq.append(
                alpha
            )

            a = alpha.unsqueeze(1)

            cand = torch.tanh(
                F.linear(
                    h,
                    wh
                )
                +
                inp[:,t,:]
            )

            h = (
                (1-a)*h
                +
                a*cand
            )

        pred = self.readout(h)

        if return_alpha:
            return (
                pred,
                torch.stack(
                    alpha_seq,
                    dim=1
                )
            )

        return pred


# ============================================================
# Train
# ============================================================

def train_model(
    seed,
    controller_type,
    train_loader,
    val_loader
):

    seed_all(seed)

    model = (
        ForecastRNN(
            controller_type
        )
        .to(DEVICE)
    )

    opt = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best = float('inf')
    best_state = None
    bad = 0
    best_epoch = 0

    for epoch in range(
        1,
        EPOCHS+1
    ):

        model.train()

        t0 = time.perf_counter()

        total = 0.0
        seen = 0

        for x,y in train_loader:

            x = x.to(
                DEVICE,
                non_blocking=PIN_MEMORY
            )

            y = y.to(
                DEVICE,
                non_blocking=PIN_MEMORY
            )

            opt.zero_grad(
                set_to_none=True
            )

            pred = model(x)

            loss = F.mse_loss(
                pred,
                y
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )

            opt.step()

            total += (
                loss.item()
                *
                x.size(0)
            )

            seen += x.size(0)

        val = evaluate(
            model,
            val_loader
        )

        elapsed = (
            time.perf_counter()
            -
            t0
        )

        print(
            f"{controller_type} | "
            f"epoch={epoch:02d} | "
            f"train={total/seen:.6f} | "
            f"val={val:.6f} | "
            f"time={elapsed:.2f}s"
        )

        if val < best - 1e-8:

            best = val
            best_epoch = epoch
            bad = 0

            best_state = {
                k:v.detach().cpu().clone()
                for k,v in model.state_dict().items()
            }

        else:

            bad += 1

        if bad >= PATIENCE:

            print(
                f"{controller_type} | early stopping"
            )

            break

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    return (
        model,
        best,
        best_epoch
    )


@torch.no_grad()
def evaluate(
    model,
    loader
):

    model.eval()

    sq = 0.0
    n = 0

    for x,y in loader:

        x = x.to(
            DEVICE,
            non_blocking=PIN_MEMORY
        )

        y = y.to(
            DEVICE,
            non_blocking=PIN_MEMORY
        )

        p = model(x)

        sq += (
            (p-y).pow(2)
        ).sum().item()

        n += y.numel()

    return sq / max(n,1)


# ============================================================
# Transition analysis
# ============================================================

@torch.no_grad()
def collect_test_trace(
    model,
    features,
    target_scaled,
    regimes
):

    model.eval()

    # Predict only on the full held-out tail.
    start = VAL_END
    end = TOTAL_POINTS

    X = []
    Y = []
    R = []
    TIDX = []

    for t in range(
        start + LOOKBACK,
        end
    ):

        X.append(
            features[
                t-LOOKBACK:t
            ]
        )

        Y.append(
            target_scaled[t]
        )

        R.append(
            regimes[t]
        )

        TIDX.append(t)

    X = torch.from_numpy(
        np.asarray(
            X,
            dtype=np.float32
        )
    ).to(
        DEVICE
    )

    pred, alpha = model(
        X,
        return_alpha=True
    )

    pred_np = (
        pred.detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    alpha_np = (
        alpha.detach()
        .cpu()
        .numpy()
    )

    y_np = np.asarray(
        Y,
        dtype=np.float32
    )

    r_np = np.asarray(
        R,
        dtype=np.int64
    )

    tidx = np.asarray(
        TIDX,
        dtype=np.int64
    )

    err = np.abs(
        pred_np-y_np
    )

    return pd.DataFrame({
        "t": tidx,
        "regime": r_np,
        "prediction": pred_np,
        "target": y_np,
        "abs_error": err,
        "alpha_final": alpha_np[:,-1]
    })


def adaptation_stats(
    trace
):

    results = []

    # Regime starts in the full series.
    test_start = VAL_END

    first_test_regime = (
        test_start // REGIME_LEN
    )

    last_test_regime = (
        (TOTAL_POINTS-1)
        // REGIME_LEN
    )

    for new_regime in range(
        first_test_regime + 1,
        last_test_regime + 1
    ):

        transition = (
            new_regime
            *
            REGIME_LEN
        )

        # Keep windows around transition.
        pre_mask = (
            (trace.t >= transition-100)
            &
            (trace.t < transition)
        )

        post_mask = (
            (trace.t >= transition)
            &
            (trace.t < transition+200)
        )

        if (
            pre_mask.sum() < 10
            or
            post_mask.sum() < 10
        ):
            continue

        pre = trace[
            pre_mask
        ]

        post = trace[
            post_mask
        ]

        pre_alpha = float(
            pre.alpha_final.mean()
        )

        post_alpha = float(
            post.alpha_final.mean()
        )

        delta_alpha = (
            post_alpha
            -
            pre_alpha
        )

        pre_error = float(
            pre.abs_error.mean()
        )

        post_error = float(
            post.abs_error.mean()
        )

        # Adaptation lag:
        # first t after transition at which a 20-point moving average
        # reaches 80% of the eventual 100-point post-transition shift.
        post_sorted = post.sort_values("t")

        eventual = float(
            post_sorted
            .iloc[
                max(
                    0,
                    len(post_sorted)-50
                ):
            ]
            .alpha_final.mean()
        )

        target_alpha = (
            pre_alpha
            +
            0.80
            *
            (
                eventual
                -
                pre_alpha
            )
        )

        lag = np.nan

        for i in range(
            0,
            max(
                1,
                len(post_sorted)-20
            )
        ):

            window = (
                post_sorted
                .iloc[i:i+20]
            )

            if len(window) < 20:
                break

            wmean = float(
                window.alpha_final.mean()
            )

            if (
                (
                    eventual
                    -
                    pre_alpha
                )
                >= 0
            ):

                reached = (
                    wmean
                    >=
                    target_alpha
                )

            else:

                reached = (
                    wmean
                    <=
                    target_alpha
                )

            if reached:

                lag = float(
                    window.t.iloc[-1]
                    -
                    transition
                )

                break

        results.append({
            "TransitionAt": transition,
            "FromRegime": new_regime-1,
            "ToRegime": new_regime,
            "PreAlpha": pre_alpha,
            "PostAlphaMean200": post_alpha,
            "DeltaAlpha": delta_alpha,
            "PreMAE100": pre_error,
            "PostMAE200": post_error,
            "AdaptationLagSteps": lag
        })

    return pd.DataFrame(
        results
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("="*80)
    print(
        "FAC-RNN REGIME-SHIFT v1"
    )
    print("="*80)

    y, ys, features, regimes, mean, std = (
        prepare_series()
    )

    print(
        "Device:",
        DEVICE
    )

    print(
        "Total points:",
        TOTAL_POINTS
    )

    print(
        "Regime length:",
        REGIME_LEN
    )

    print(
        "Train/Val/Test:",
        TRAIN_END,
        VAL_END,
        TOTAL_POINTS
    )

    print(
        "Test begins at regime:",
        int(
            VAL_END // REGIME_LEN
        )
    )

    Xtr,Ytr = build_windows(
        features,
        ys,
        0,
        TRAIN_END,
        TRAIN_WINDOWS
    )

    Xva,Yva = build_windows(
        features,
        ys,
        TRAIN_END,
        VAL_END,
        VAL_WINDOWS
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(Xtr),
            torch.from_numpy(Ytr)
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=PIN_MEMORY,
        num_workers=0
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(Xva),
            torch.from_numpy(Yva)
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=PIN_MEMORY,
        num_workers=0
    )

    all_results = []
    all_transition = []

    for typ in [
        "fixed",
        "nonfuzzy",
        "fuzzy"
    ]:

        model, best_val, best_epoch = (
            train_model(
                SEED,
                typ,
                train_loader,
                val_loader
            )
        )

        trace = collect_test_trace(
            model,
            features,
            ys,
            regimes
        )

        test_mse = float(
            (
                trace.abs_error
                **2
            ).mean()
        )

        test_mae = float(
            trace.abs_error.mean()
        )

        row = {
            "Controller": typ,
            "BestEpoch": best_epoch,
            "BestValMSE": best_val,
            "TestMSE": test_mse,
            "TestMAE": test_mae,
            "Kappa": KAPPA,
            "AlphaMin": ALPHA_MIN
        }

        if typ == "fuzzy":

            rules = (
                model.ctrl
                .rule_values()
                .detach()
                .cpu()
                .numpy()
            )

            print(
                "Learned fuzzy rules:",
                [float(v) for v in rules]
            )

            for i,v in enumerate(
                rules,
                1
            ):
                row[
                    f"AlphaRule{i}"
                ] = float(v)

        all_results.append(
            row
        )

        trans = adaptation_stats(
            trace
        )

        trans[
            "Controller"
        ] = typ

        all_transition.append(
            trans
        )

        # Save trace per controller.
        trace.to_csv(
            f"fac_regimeshift_trace_{typ}.csv",
            index=False
        )

    results_df = pd.DataFrame(
        all_results
    )

    transition_df = pd.concat(
        all_transition,
        ignore_index=True
    )

    print()
    print("="*80)
    print(
        "REGIME-SHIFT MODEL SUMMARY"
    )
    print("="*80)

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print("="*80)
    print(
        "TRANSITION ADAPTATION SUMMARY"
    )
    print("="*80)

    print(
        transition_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Compare fuzzy against non-fuzzy.
    # --------------------------------------------------------

    print()
    print(
        "Fuzzy vs Non-Fuzzy test MSE reduction (%):"
    )

    f = float(
        results_df.loc[
            results_df.Controller=="fuzzy",
            "TestMSE"
        ].iloc[0]
    )

    n = float(
        results_df.loc[
            results_df.Controller=="nonfuzzy",
            "TestMSE"
        ].iloc[0]
    )

    print(
        100.0*(n-f)/n
    )

    results_df.to_csv(
        "fac_regimeshift_results.csv",
        index=False
    )

    transition_df.to_csv(
        "fac_regimeshift_transitions.csv",
        index=False
    )

    print()
    print("Saved:")
    print(
        "  fac_regimeshift_results.csv"
    )
    print(
        "  fac_regimeshift_transitions.csv"
    )
    print(
        "  fac_regimeshift_trace_fixed.csv"
    )
    print(
        "  fac_regimeshift_trace_nonfuzzy.csv"
    )
    print(
        "  fac_regimeshift_trace_fuzzy.csv"
    )
