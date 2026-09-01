
# ============================================================
# FAC-RNN FORECASTING v2 — FUZZY TIME-SCALE MECHANISM ANALYSIS
#
# Purpose:
#   Analyze whether the learned fuzzy controller actually adapts
#   its time scale to the underlying dynamical regime.
#
# We DO NOT change the forecasting model or benchmark.
#
# Questions:
#   1) Does alpha_t differ across the three true regimes?
#   2) Does alpha_t increase with local temporal change |delta_t|?
#   3) Is alpha_t associated with regime transitions?
#   4) How stable is the learned alpha policy across 3 seeds?
#
# Models:
#   Only the proposed Fuzzy Adaptive alpha RNN is analyzed here.
#
# Data:
#   Same deterministic switching nonlinear process as forecasting v1.
#
# Regimes:
#   0 = slow nonlinear oscillator
#   1 = faster nonlinear oscillator
#   2 = nonlinear autoregressive regime
#
# Chronological split:
#   60% train
#   20% validation
#   20% test
#
# The test sequence is NOT used during training.
#
# Outputs:
#   fac_forecasting_alpha_analysis.csv
#   fac_forecasting_alpha_regime_summary.csv
#   fac_forecasting_alpha_seed_summary.csv
#   fac_forecasting_alpha_examples.csv
#
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

SEEDS = [42, 123, 456]

TOTAL_POINTS = 24000
LOOKBACK = 64

TRAIN_FRAC = 0.60
VAL_FRAC = 0.20

TRAIN_WINDOWS = 9000
VAL_WINDOWS = 2500
TEST_WINDOWS = 2500

BATCH_SIZE = 128
HIDDEN = 48

EPOCHS = 15
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


# ============================================================
# Seed
# ============================================================

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Data generation WITH regime labels
# ============================================================

def generate_series_and_regimes(
    total_points=TOTAL_POINTS,
    seed=2026
):
    rng = np.random.default_rng(seed)

    y = np.zeros(
        total_points,
        dtype=np.float64
    )

    regimes = np.zeros(
        total_points,
        dtype=np.int64
    )

    y[0] = 0.1
    y[1] = 0.12

    regime_len = 800

    for t in range(
        2,
        total_points
    ):

        regime = (
            (t // regime_len) % 3
        )

        regimes[t] = regime

        noise = (
            NOISE_STD
            *
            rng.standard_normal()
        )

        if regime == 0:

            y[t] = (
                0.94 * y[t - 1]
                -
                0.08 * y[t - 2]
                +
                0.10
                *
                np.sin(
                    0.055 * t
                    +
                    0.7 * y[t - 1]
                )
                +
                noise
            )

        elif regime == 1:

            y[t] = (
                0.72 * y[t - 1]
                +
                0.12 * y[t - 2]
                +
                0.22
                *
                np.sin(
                    0.16 * t
                    +
                    1.2 * y[t - 1]
                )
                +
                noise
            )

        else:

            y[t] = (
                0.62 * y[t - 1]
                +
                0.08 * y[t - 2]
                -
                0.10 * y[t - 1] ** 3
                +
                0.14
                *
                np.sin(
                    0.095 * t
                    +
                    0.8 * y[t - 1]
                )
                +
                noise
            )

    return (
        y.astype(np.float32),
        regimes
    )


def make_features(
    y,
    mean,
    std
):
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

    X = np.stack(
        [
            ys,
            delta
        ],
        axis=-1
    ).astype(
        np.float32
    )

    return X


def build_windows(
    features,
    target,
    regimes,
    start,
    end,
    lookback,
    max_windows
):
    valid_start = (
        start + lookback
    )

    valid_end = min(
        end,
        valid_start + max_windows
    )

    X = []
    Y = []
    R = []
    E = []

    for t in range(
        valid_start,
        valid_end
    ):

        X.append(
            features[
                t - lookback:t
            ]
        )

        Y.append(
            target[t]
        )

        # Regime of the forecast target.
        R.append(
            regimes[t]
        )

        # |delta| of the forecast target location.
        E.append(
            abs(
                features[t, 1]
            )
        )

    return (
        np.asarray(
            X,
            dtype=np.float32
        ),
        np.asarray(
            Y,
            dtype=np.float32
        ).reshape(-1, 1),
        np.asarray(
            R,
            dtype=np.int64
        ),
        np.asarray(
            E,
            dtype=np.float32
        )
    )


def prepare_data():

    y, regimes = (
        generate_series_and_regimes()
    )

    train_end = int(
        TOTAL_POINTS
        *
        TRAIN_FRAC
    )

    val_end = int(
        TOTAL_POINTS
        *
        (
            TRAIN_FRAC
            +
            VAL_FRAC
        )
    )

    train_y = y[
        :train_end
    ]

    mean = float(
        train_y.mean()
    )

    std = float(
        train_y.std()
    )

    features = make_features(
        y,
        mean,
        std
    )

    target = (
        (
            y
            -
            mean
        )
        /
        max(
            std,
            1e-8
        )
    ).astype(
        np.float32
    )

    tr = build_windows(
        features,
        target,
        regimes,
        0,
        train_end,
        LOOKBACK,
        TRAIN_WINDOWS
    )

    va = build_windows(
        features,
        target,
        regimes,
        train_end,
        val_end,
        LOOKBACK,
        VAL_WINDOWS
    )

    te = build_windows(
        features,
        target,
        regimes,
        val_end,
        TOTAL_POINTS,
        LOOKBACK,
        TEST_WINDOWS
    )

    return (
        tr,
        va,
        te,
        y,
        regimes,
        mean,
        std
    )


def loader_from_tuple(
    data_tuple,
    shuffle
):
    X, Y, R, E = data_tuple

    return DataLoader(
        TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(Y),
            torch.from_numpy(R),
            torch.from_numpy(E)
        ),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        pin_memory=PIN_MEMORY,
        num_workers=0
    )


# ============================================================
# Model components
# ============================================================

class ContractiveMatrix(nn.Module):

    def __init__(
        self,
        hidden
    ):
        super().__init__()

        self.raw = nn.Parameter(
            torch.empty(
                hidden,
                hidden
            )
        )

        nn.init.orthogonal_(
            self.raw
        )

    def forward(self):

        fro = self.raw.norm(
            p="fro"
        ).clamp_min(
            1e-12
        )

        return (
            self.raw
            *
            torch.clamp(
                KAPPA / fro,
                max=1.0
            )
        )


class FuzzyAlphaController(nn.Module):

    def __init__(self):

        super().__init__()

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

        for layer in self.score:

            if isinstance(
                layer,
                nn.Linear
            ):

                nn.init.xavier_uniform_(
                    layer.weight
                )

                nn.init.zeros_(
                    layer.bias
                )

        alpha_init = torch.tensor(
            [
                0.02,
                0.10,
                0.90
            ]
        )

        p = (
            (
                alpha_init
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
            1.0 - 1e-5
        )

        self.rule_logits = nn.Parameter(
            torch.log(
                p
                /
                (
                    1.0
                    -
                    p
                )
            )
        )

        self.register_buffer(
            "centers",
            torch.tensor(
                [-1.0, 0.0, 1.0]
            )
        )

        self.log_sigma = nn.Parameter(
            torch.tensor(
                -0.5
            )
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

    def forward(
        self,
        x
    ):

        z = torch.tanh(
            self.score(
                x
            ).squeeze(-1)
        )

        diff = (
            z.unsqueeze(-1)
            -
            self.centers.view(
                1,
                3
            )
        )

        sigma = (
            F.softplus(
                self.log_sigma
            )
            +
            0.05
        )

        membership = torch.exp(
            -diff.pow(2)
            /
            (
                2.0
                *
                sigma.pow(2)
            )
        )

        weights = (
            membership
            /
            membership.sum(
                dim=-1,
                keepdim=True
            ).clamp_min(
                1e-8
            )
        )

        rules = self.rule_values()

        alpha = (
            weights
            *
            rules.view(
                1,
                3
            )
        ).sum(
            dim=-1
        )

        return (
            alpha,
            weights
        )


class FACForecastRNN(nn.Module):

    def __init__(self):

        super().__init__()

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
            torch.zeros(
                HIDDEN
            )
        )

        nn.init.xavier_uniform_(
            self.W_x
        )

        self.controller = (
            FuzzyAlphaController()
        )

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

        W_h = self.W_h()

        input_part = (
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

        alpha_sequence = []

        for t in range(
            T
        ):

            xt = x[:, t, :]

            alpha, _ = (
                self.controller(
                    xt
                )
            )

            alpha = alpha.unsqueeze(
                1
            )

            candidate = torch.tanh(
                F.linear(
                    h,
                    W_h
                )
                +
                input_part[:, t, :]
            )

            h = (
                (1.0 - alpha)
                *
                h
                +
                alpha
                *
                candidate
            )

            if return_alpha:

                alpha_sequence.append(
                    alpha.squeeze(1)
                )

        pred = self.readout(
            h
        )

        if return_alpha:

            return (
                pred,
                torch.stack(
                    alpha_sequence,
                    dim=1
                )
            )

        return pred


# ============================================================
# Train
# ============================================================

def train_one(
    seed,
    train_loader,
    val_loader
):

    seed_all(seed)

    model = (
        FACForecastRNN()
        .to(DEVICE)
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0

    history = []

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        total_loss = 0.0
        seen = 0

        for x, y, _, _ in train_loader:

            x = x.to(
                DEVICE,
                non_blocking=PIN_MEMORY
            )

            y = y.to(
                DEVICE,
                non_blocking=PIN_MEMORY
            )

            optimizer.zero_grad(
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

            optimizer.step()

            total_loss += (
                loss.item()
                *
                x.size(0)
            )

            seen += x.size(0)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            -
            t0
        )

        train_mse = (
            total_loss
            /
            max(
                seen,
                1
            )
        )

        val_mse = (
            evaluate(
                model,
                val_loader
            )
        )

        history.append({
            "seed": seed,
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "epoch_time_sec": elapsed
        })

        print(
            f"Seed {seed} | "
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train={train_mse:.6f} | "
            f"val={val_mse:.6f} | "
            f"time={elapsed:.2f}s"
        )

        if val_mse < (
            best_val
            -
            1e-8
        ):

            best_val = val_mse
            best_epoch = epoch
            bad = 0

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        else:

            bad += 1

        if bad >= PATIENCE:

            print(
                "Early stopping."
            )

            break

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return (
        model,
        best_val,
        best_epoch,
        history
    )


def evaluate(
    model,
    loader
):

    model.eval()

    sq = 0.0
    n = 0

    with torch.no_grad():

        for x, y, _, _ in loader:

            x = x.to(
                DEVICE,
                non_blocking=PIN_MEMORY
            )

            y = y.to(
                DEVICE,
                non_blocking=PIN_MEMORY
            )

            pred = model(x)

            sq += (
                (
                    pred - y
                ).pow(2)
                .sum()
                .item()
            )

            n += y.numel()

    return sq / max(n, 1)


# ============================================================
# Alpha analysis
# ============================================================

@torch.no_grad()
def analyze_model(
    model,
    test_loader
):

    model.eval()

    rows = []

    for x, y, regime, abs_delta in test_loader:

        x_device = x.to(
            DEVICE,
            non_blocking=PIN_MEMORY
        )

        pred, alpha_seq = model(
            x_device,
            return_alpha=True
        )

        # Each test window contributes the alpha trajectory.
        # We align the final alpha with the forecast target.
        final_alpha = (
            alpha_seq[:, -1]
            .detach()
            .cpu()
            .numpy()
        )

        pred_np = (
            pred
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        y_np = (
            y
            .numpy()
            .reshape(-1)
        )

        regime_np = (
            regime
            .numpy()
            .reshape(-1)
        )

        delta_np = (
            abs_delta
            .numpy()
            .reshape(-1)
        )

        err_np = np.abs(
            pred_np
            -
            y_np
        )

        for i in range(
            len(final_alpha)
        ):

            rows.append({
                "regime": int(
                    regime_np[i]
                ),
                "alpha": float(
                    final_alpha[i]
                ),
                "abs_delta": float(
                    delta_np[i]
                ),
                "abs_error": float(
                    err_np[i]
                )
            })

    df = pd.DataFrame(
        rows
    )

    return df


def fit_linear_corr(
    x,
    y
):

    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) < 3:
        return float("nan")

    if np.std(x) < 1e-12:
        return float("nan")

    if np.std(y) < 1e-12:
        return float("nan")

    return float(
        np.corrcoef(
            x,
            y
        )[0, 1]
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print(
        "FAC-RNN FORECASTING v2 — ALPHA MECHANISM ANALYSIS"
    )
    print("=" * 80)

    (
        train_data,
        val_data,
        test_data,
        y,
        regimes,
        mean,
        std
    ) = prepare_data()

    print(
        "Device:",
        DEVICE
    )

    print(
        "Train / Val / Test:",
        len(train_data[0]),
        len(val_data[0]),
        len(test_data[0])
    )

    print(
        "Lookback:",
        LOOKBACK
    )

    print(
        "Regime length:",
        800
    )

    print(
        "Chronological split; test remains unseen."
    )

    all_seed_summary = []
    all_regime_rows = []
    all_example_rows = []
    all_history = []

    for seed in SEEDS:

        print()
        print("#" * 80)
        print(
            f"START SEED {seed}"
        )
        print(
            "#" * 80
        )

        train_loader = loader_from_tuple(
            train_data,
            True
        )

        val_loader = loader_from_tuple(
            val_data,
            False
        )

        test_loader = loader_from_tuple(
            test_data,
            False
        )

        model, best_val, best_epoch, history = (
            train_one(
                seed,
                train_loader,
                val_loader
            )
        )

        all_history.extend(
            history
        )

        test_mse = evaluate(
            model,
            test_loader
        )

        alpha_df = analyze_model(
            model,
            test_loader
        )

        # ----------------------------------------------------
        # Overall alpha statistics
        # ----------------------------------------------------

        alpha_mean = float(
            alpha_df["alpha"].mean()
        )

        alpha_std = float(
            alpha_df["alpha"].std()
        )

        delta_corr = fit_linear_corr(
            alpha_df["abs_delta"],
            alpha_df["alpha"]
        )

        error_corr = fit_linear_corr(
            alpha_df["alpha"],
            alpha_df["abs_error"]
        )

        # ----------------------------------------------------
        # Per-regime statistics
        # ----------------------------------------------------

        print()
        print(
            f"Seed {seed} | "
            f"Test MSE={test_mse:.8f}"
        )

        for regime in [
            0,
            1,
            2
        ]:

            sub = alpha_df[
                alpha_df["regime"]
                ==
                regime
            ]

            alpha_r = (
                sub["alpha"]
            )

            delta_r = (
                sub["abs_delta"]
            )

            error_r = (
                sub["abs_error"]
            )

            regime_mean = float(
                alpha_r.mean()
            )

            regime_std = float(
                alpha_r.std()
            )

            regime_delta = float(
                delta_r.mean()
            )

            regime_error = float(
                error_r.mean()
            )

            regime_corr = fit_linear_corr(
                delta_r,
                alpha_r
            )

            print(
                f"Regime {regime}: "
                f"alpha={regime_mean:.6f}±{regime_std:.6f} | "
                f"|delta|={regime_delta:.6f} | "
                f"MAE={regime_error:.6f} | "
                f"corr(alpha,|delta|)={regime_corr:.4f}"
            )

            all_regime_rows.append({
                "Seed": seed,
                "Regime": regime,
                "Samples": len(sub),
                "AlphaMean": regime_mean,
                "AlphaStd": regime_std,
                "AbsDeltaMean": regime_delta,
                "AbsErrorMean": regime_error,
                "CorrAlphaAbsDelta": regime_corr
            })

        # ----------------------------------------------------
        # Representative samples
        # ----------------------------------------------------

        representative = alpha_df.iloc[
            np.linspace(
                0,
                len(alpha_df) - 1,
                min(
                    200,
                    len(alpha_df)
                ),
                dtype=int
            )
        ].copy()

        representative[
            "Seed"
        ] = seed

        representative[
            "Index"
        ] = np.arange(
            len(representative)
        )

        all_example_rows.extend(
            representative.to_dict(
                orient="records"
            )
        )

        # ----------------------------------------------------
        # Seed-level summary
        # ----------------------------------------------------

        all_seed_summary.append({
            "Seed": seed,
            "BestEpoch": best_epoch,
            "BestValMSE": best_val,
            "TestMSE": test_mse,
            "AlphaMean": alpha_mean,
            "AlphaStd": alpha_std,
            "CorrAlphaAbsDelta": delta_corr,
            "CorrAlphaAbsError": error_corr,
            "Kappa": KAPPA,
            "AlphaMin": ALPHA_MIN,
            "AlphaMax": ALPHA_MAX,
            "ContractionQBound":
                1.0
                -
                ALPHA_MIN
                *
                (
                    1.0
                    -
                    KAPPA
                )
        })

        # Incremental save.
        pd.DataFrame(
            all_seed_summary
        ).to_csv(
            "fac_forecasting_alpha_seed_summary.csv",
            index=False
        )

        pd.DataFrame(
            all_regime_rows
        ).to_csv(
            "fac_forecasting_alpha_regime_summary.csv",
            index=False
        )

        pd.DataFrame(
            all_example_rows
        ).to_csv(
            "fac_forecasting_alpha_examples.csv",
            index=False
        )

    # ========================================================
    # Aggregate
    # ========================================================

    seed_df = pd.DataFrame(
        all_seed_summary
    )

    regime_df = pd.DataFrame(
        all_regime_rows
    )

    example_df = pd.DataFrame(
        all_example_rows
    )

    history_df = pd.DataFrame(
        all_history
    )

    aggregate_regime = (
        regime_df
        .groupby(
            "Regime",
            as_index=False
        )
        .agg(
            Seeds=("Seed", "count"),
            AlphaMean=("AlphaMean", "mean"),
            AlphaStdAcrossSeeds=("AlphaMean", "std"),
            MeanWithinSeedAlphaStd=(
                "AlphaStd",
                "mean"
            ),
            AbsDeltaMean=("AbsDeltaMean", "mean"),
            AbsErrorMean=("AbsErrorMean", "mean"),
            CorrAlphaAbsDeltaMean=(
                "CorrAlphaAbsDelta",
                "mean"
            )
        )
    )

    print()
    print("=" * 80)
    print(
        "FINAL ALPHA MECHANISM SUMMARY"
    )
    print("=" * 80)

    print(
        seed_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Aggregate regime behavior:"
    )

    print(
        aggregate_regime.to_string(
            index=False
        )
    )

    print()
    print(
        "Across-seed Test MSE:"
    )

    print(
        f"mean = {seed_df['TestMSE'].mean():.8f}"
    )

    print(
        f"std  = {seed_df['TestMSE'].std():.8f}"
    )

    print()
    print(
        "Across-seed alpha mean:"
    )

    print(
        f"mean = {seed_df['AlphaMean'].mean():.8f}"
    )

    print(
        f"std  = {seed_df['AlphaMean'].std():.8f}"
    )

    print()
    print(
        "Files saved:"
    )

    seed_df.to_csv(
        "fac_forecasting_alpha_seed_summary.csv",
        index=False
    )

    aggregate_regime.to_csv(
        "fac_forecasting_alpha_regime_summary.csv",
        index=False
    )

    example_df.to_csv(
        "fac_forecasting_alpha_examples.csv",
        index=False
    )

    history_df.to_csv(
        "fac_forecasting_alpha_history.csv",
        index=False
    )

    # Full per-window observations reconstructed from examples
    example_df.to_csv(
        "fac_forecasting_alpha_analysis.csv",
        index=False
    )

    print(
        "  fac_forecasting_alpha_analysis.csv"
    )

    print(
        "  fac_forecasting_alpha_regime_summary.csv"
    )

    print(
        "  fac_forecasting_alpha_seed_summary.csv"
    )

    print(
        "  fac_forecasting_alpha_examples.csv"
    )

    print(
        "  fac_forecasting_alpha_history.csv"
    )
