
# ============================================================
# FAC-RNN FORECASTING v1 — ADAPTIVE TIME-SCALE FOR TIME SERIES
#
# Purpose:
#   First forecasting benchmark for FAC-RNN after the Adding and
#   Copy-Memory experiments.
#
# Research question:
#   Does fuzzy adaptive time-scale improve one-step forecasting
#   for a nonlinear time series with changing local dynamics?
#
# Model:
#
#   h_t = (1-alpha_t) h_(t-1)
#         + alpha_t tanh(W_h h_(t-1) + W_x z_t + b)
#
# where:
#
#   z_t = [y_t, delta_t]
#   delta_t = y_t - y_(t-1)
#
# The controller sees exactly the same observable information
# in the fixed/non-fuzzy/fuzzy comparisons.
#
# Stability design:
#
#   ||W_h||_2 <= ||W_h||_F <= KAPPA < 1
#   alpha_t >= ALPHA_MIN > 0
#
# Therefore:
#
#   L_t <= 1 - alpha_min*(1-KAPPA) < 1
#
# This script measures:
#   - Test MSE
#   - Test MAE
#   - RMSE
#   - parameter count
#   - training time
#
# Data:
#   A deterministic nonlinear switching-regime series is generated
#   once and split chronologically:
#
#       60% train
#       20% validation
#       20% test
#
# Windows are created AFTER the chronological split boundaries,
# preventing future leakage.
#
# Forecasting task:
#   given LOOKBACK observations, predict the next value.
#
# Multi-seed replication:
#   SEEDS = [42,123,456]
#
# IMPORTANT:
#   This is a research prototype. It does not establish general
#   superiority. It establishes whether the proposed mechanism
#   deserves continued forecasting experiments.
# ============================================================

import time
import random
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

# Stability design.
KAPPA = 0.90
ALPHA_MIN = 0.02
ALPHA_MAX = 1.00

# Small observation noise.
NOISE_STD = 0.03

INPUT_DIM = 2  # [y_t, delta_t]


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
# Reproducibility
# ============================================================

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Synthetic switching nonlinear series
# ============================================================

def generate_series(
    total_points=TOTAL_POINTS,
    seed=2026
):
    """
    One deterministic series with alternating dynamical regimes.

    Regime A:
        slowly varying nonlinear oscillator.

    Regime B:
        faster nonlinear oscillator.

    Regime C:
        damped autoregressive nonlinear regime.

    Regime boundaries are NOT supplied to the model.

    This gives the time series changing local time scales, which is
    exactly the phenomenon the adaptive-alpha mechanism is intended
    to address.
    """

    rng = np.random.default_rng(seed)

    y = np.zeros(total_points, dtype=np.float64)

    # Initial state.
    y[0] = 0.1
    y[1] = 0.12

    # Regime length.
    regime_len = 800

    for t in range(2, total_points):

        regime = (
            (t // regime_len) % 3
        )

        noise = (
            NOISE_STD
            * rng.standard_normal()
        )

        if regime == 0:
            # Slow nonlinear oscillatory dynamics.
            y[t] = (
                0.94 * y[t - 1]
                - 0.08 * y[t - 2]
                + 0.10 * np.sin(
                    0.055 * t
                    + 0.7 * y[t - 1]
                )
                + noise
            )

        elif regime == 1:
            # Faster nonlinear regime.
            y[t] = (
                0.72 * y[t - 1]
                + 0.12 * y[t - 2]
                + 0.22 * np.sin(
                    0.16 * t
                    + 1.2 * y[t - 1]
                )
                + noise
            )

        else:
            # Stronger nonlinear autoregressive regime.
            y[t] = (
                0.62 * y[t - 1]
                + 0.08 * y[t - 2]
                - 0.10 * y[t - 1] ** 3
                + 0.14 * np.sin(
                    0.095 * t
                    + 0.8 * y[t - 1]
                )
                + noise
            )

    # Standardize globally only using the full synthetic generating
    # process is acceptable for a synthetic controlled experiment,
    # but the forecast split itself remains chronological.
    # To avoid even this shortcut, standardize using training only
    # later in prepare_splits.
    return y.astype(np.float32)


# ============================================================
# Chronological split + window preparation
# ============================================================

def make_features(y, mean, std):
    """
    Observable input:
        value
        first difference

    Standardization parameters are supplied externally and are
    fitted on training observations only.
    """

    y_scaled = (
        y - mean
    ) / max(std, 1e-8)

    delta = np.zeros_like(
        y_scaled
    )

    delta[1:] = (
        y_scaled[1:]
        -
        y_scaled[:-1]
    )

    return np.stack(
        [
            y_scaled,
            delta
        ],
        axis=-1
    ).astype(
        np.float32
    )


def build_windows(
    feature_array,
    target_array,
    start,
    end,
    lookback,
    max_windows
):
    """
    Windows are constrained inside [start,end).

    x = observations [t-lookback, ..., t-1]
    y = observation t
    """

    valid_start = (
        start + lookback
    )

    valid_end = min(
        end,
        valid_start + max_windows
    )

    X = []
    Y = []

    for t in range(
        valid_start,
        valid_end
    ):

        X.append(
            feature_array[
                t - lookback:t
            ]
        )

        Y.append(
            target_array[t]
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(Y, dtype=np.float32).reshape(-1, 1)
    )


def prepare_data():
    y = generate_series()

    train_end = int(
        TOTAL_POINTS * TRAIN_FRAC
    )

    val_end = int(
        TOTAL_POINTS
        *
        (TRAIN_FRAC + VAL_FRAC)
    )

    # Fit scaling ONLY on the training target observations.
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

    # Raw target standardized with training statistics.
    target = (
        (y - mean)
        /
        max(std, 1e-8)
    ).astype(
        np.float32
    )

    # IMPORTANT:
    # Validation/test windows begin only within their own segment.
    Xtr, Ytr = build_windows(
        features,
        target,
        0,
        train_end,
        LOOKBACK,
        TRAIN_WINDOWS
    )

    Xva, Yva = build_windows(
        features,
        target,
        train_end,
        val_end,
        LOOKBACK,
        VAL_WINDOWS
    )

    Xte, Yte = build_windows(
        features,
        target,
        val_end,
        TOTAL_POINTS,
        LOOKBACK,
        TEST_WINDOWS
    )

    print(
        "Chronological split:",
        f"train=[0,{train_end})",
        f"val=[{train_end},{val_end})",
        f"test=[{val_end},{TOTAL_POINTS})"
    )

    print(
        "Window shapes:",
        Xtr.shape,
        Xva.shape,
        Xte.shape
    )

    return (
        Xtr,
        Ytr,
        Xva,
        Yva,
        Xte,
        Yte,
        mean,
        std
    )


def make_loader(
    X,
    Y,
    shuffle
):
    return DataLoader(
        TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(Y)
        ),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        pin_memory=PIN_MEMORY,
        num_workers=0
    )


# ============================================================
# Contractive matrix
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

        scale = torch.clamp(
            KAPPA / fro,
            max=1.0
        )

        return (
            scale
            * self.raw
        )


# ============================================================
# Controllers
# ============================================================

class FixedAlphaController(nn.Module):

    def __init__(
        self,
        alpha=0.10
    ):
        super().__init__()

        self.register_buffer(
            "alpha",
            torch.tensor(
                float(alpha)
            )
        )

    def forward(
        self,
        x
    ):
        return self.alpha.expand(
            x.shape[:-1]
        )


class LearnedAlphaController(nn.Module):

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

        for layer in self.net:

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

    def forward(
        self,
        x
    ):

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
            torch.sigmoid(
                z
            )
        )


class FuzzyAlphaController(nn.Module):
    """
    Three-rule Sugeno-style fuzzy time-scale controller.

    A scalar observable salience coordinate is constructed from:
        y_t
        |delta_t|

    The controller outputs alpha in a guaranteed interval.
    """

    def __init__(self):

        super().__init__()

        # Small affine projection of the observable state.
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

        return alpha


# ============================================================
# Forecasting model
# ============================================================

class FACForecastRNN(nn.Module):

    def __init__(
        self,
        controller_type
    ):

        super().__init__()

        self.controller_type = (
            controller_type
        )

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

        if controller_type == "fixed":

            self.controller = (
                FixedAlphaController()
            )

        elif controller_type == "nonfuzzy":

            self.controller = (
                LearnedAlphaController()
            )

        elif controller_type == "fuzzy":

            self.controller = (
                FuzzyAlphaController()
            )

        else:

            raise ValueError(
                "Invalid controller_type"
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
        x
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

        for t in range(
            T
        ):

            xt = x[:, t, :]

            alpha = self.controller(
                xt
            ).unsqueeze(
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

        return self.readout(
            h
        )


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader
):

    model.eval()

    sq = 0.0
    ab = 0.0
    n = 0

    for x, y in loader:

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

        ab += (
            (
                pred - y
            ).abs()
            .sum()
            .item()
        )

        n += y.numel()

    mse = sq / n
    mae = ab / n

    return {
        "mse": mse,
        "mae": mae,
        "rmse": math.sqrt(mse)
    }


# ============================================================
# Train one model
# ============================================================

def run_one(
    seed,
    name,
    controller_type,
    train_loader,
    val_loader,
    test_loader
):

    seed_all(seed)

    model = (
        FACForecastRNN(
            controller_type
        )
        .to(DEVICE)
    )

    params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    q_bound = (
        1.0
        -
        ALPHA_MIN
        *
        (
            1.0
            -
            KAPPA
        )
    )

    print()
    print("=" * 80)
    print(
        f"Seed={seed} | "
        f"{name} | "
        f"parameters={params}"
    )
    print("=" * 80)

    print(
        f"Stability design: "
        f"||W_h||2 <= {KAPPA:.3f}, "
        f"alpha >= {ALPHA_MIN:.3f}, "
        f"q <= {q_bound:.6f}"
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

        loss_sum = 0.0
        seen = 0

        for x, y in train_loader:

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

            loss_sum += (
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
            loss_sum
            /
            max(
                seen,
                1
            )
        )

        val = evaluate(
            model,
            val_loader
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_mse={train_mse:.6f} | "
            f"val_mse={val['mse']:.6f} | "
            f"val_mae={val['mae']:.6f} | "
            f"time={elapsed:.2f}s"
        )

        history.append({
            "Seed": seed,
            "Architecture": name,
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val["mse"],
            "val_mae": val["mae"],
            "epoch_time_sec": elapsed
        })

        if val["mse"] < (
            best_val
            -
            1e-8
        ):

            best_val = val["mse"]
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

    test = evaluate(
        model,
        test_loader
    )

    mean_epoch = float(
        np.mean(
            [
                h["epoch_time_sec"]
                for h in history
            ]
        )
    )

    result = {
        "Seed": seed,
        "Architecture": name,
        "Controller": controller_type,
        "Parameters": params,
        "BestEpoch": best_epoch,
        "BestValMSE": best_val,
        "TestMSE": test["mse"],
        "TestMAE": test["mae"],
        "TestRMSE": test["rmse"],
        "MeanEpochTimeSec": mean_epoch,
        "Kappa": KAPPA,
        "AlphaMin": ALPHA_MIN,
        "AlphaMax": ALPHA_MAX,
        "ContractionQBound": q_bound
    }

    if controller_type == "fuzzy":

        with torch.no_grad():

            rules = (
                model.controller
                .rule_values()
                .detach()
                .cpu()
                .numpy()
            )

        print(
            "Learned fuzzy alpha rules:",
            [
                float(v)
                for v in rules
            ]
        )

        for i, value in enumerate(
            rules,
            1
        ):
            result[
                f"AlphaRule{i}"
            ] = float(value)

    print(
        f"FINAL Seed={seed} | "
        f"{name} | "
        f"Test MSE={test['mse']:.8f} | "
        f"Test MAE={test['mae']:.8f} | "
        f"RMSE={test['rmse']:.8f} | "
        f"MeanEpoch={mean_epoch:.2f}s"
    )

    return result, history


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    import math

    print("=" * 80)
    print(
        "FAC-RNN FORECASTING v1"
    )
    print("=" * 80)

    print(
        "Device:",
        DEVICE
    )

    print(
        "Seeds:",
        SEEDS
    )

    print(
        "Total points:",
        TOTAL_POINTS
    )

    print(
        "Lookback:",
        LOOKBACK
    )

    print(
        "Target: one-step-ahead forecasting"
    )

    (
        Xtr,
        Ytr,
        Xva,
        Yva,
        Xte,
        Yte,
        scale_mean,
        scale_std
    ) = prepare_data()

    train_loader = make_loader(
        Xtr,
        Ytr,
        True
    )

    val_loader = make_loader(
        Xva,
        Yva,
        False
    )

    test_loader = make_loader(
        Xte,
        Yte,
        False
    )

    all_results = []
    all_history = []

    experiments = [
        (
            "A: Fixed-alpha Contractive RNN",
            "fixed"
        ),
        (
            "B: Learned Non-Fuzzy alpha RNN",
            "nonfuzzy"
        ),
        (
            "C: Fuzzy Adaptive alpha RNN",
            "fuzzy"
        )
    ]

    for seed in SEEDS:

        print()
        print("#" * 80)
        print(
            f"START SEED {seed}"
        )
        print(
            "#" * 80
        )

        for name, controller_type in experiments:

            result, history = run_one(
                seed,
                name,
                controller_type,
                train_loader,
                val_loader,
                test_loader
            )

            all_results.append(
                result
            )

            all_history.extend(
                history
            )

            # Incremental saving.
            pd.DataFrame(
                all_results
            ).to_csv(
                "fac_forecasting_results.csv",
                index=False
            )

            pd.DataFrame(
                all_history
            ).to_csv(
                "fac_forecasting_history.csv",
                index=False
            )

    results_df = pd.DataFrame(
        all_results
    )

    history_df = pd.DataFrame(
        all_history
    )

    summary_df = (
        results_df
        .groupby(
            [
                "Architecture",
                "Controller"
            ],
            as_index=False
        )
        .agg(
            Seeds=("Seed", "count"),
            TestMSEMean=("TestMSE", "mean"),
            TestMSEStd=("TestMSE", "std"),
            TestMAEMean=("TestMAE", "mean"),
            TestMAEStd=("TestMAE", "std"),
            TestRMSEMean=("TestRMSE", "mean"),
            TestRMSEStd=("TestRMSE", "std"),
            BestValMSEMean=("BestValMSE", "mean"),
            BestValMSEStd=("BestValMSE", "std"),
            MeanEpochTimeSec=(
                "MeanEpochTimeSec",
                "mean"
            ),
            Parameters=("Parameters", "first")
        )
    )

    # Pairwise improvement percentages.
    pivot = results_df.pivot(
        index="Seed",
        columns="Controller",
        values="TestMSE"
    )

    if (
        "fuzzy" in pivot.columns
        and
        "nonfuzzy" in pivot.columns
    ):

        pivot[
            "Fuzzy_MSE_Reduction_vs_NonFuzzy_pct"
        ] = (
            100.0
            *
            (
                pivot["nonfuzzy"]
                -
                pivot["fuzzy"]
            )
            /
            pivot["nonfuzzy"]
        )

    if (
        "fuzzy" in pivot.columns
        and
        "fixed" in pivot.columns
    ):

        pivot[
            "Fuzzy_MSE_Reduction_vs_Fixed_pct"
        ] = (
            100.0
            *
            (
                pivot["fixed"]
                -
                pivot["fuzzy"]
            )
            /
            pivot["fixed"]
        )

    advantage_df = pivot.reset_index()

    print()
    print("=" * 80)
    print(
        "FINAL FORECASTING MULTI-SEED SUMMARY"
    )
    print("=" * 80)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print(
        "PER-SEED FUZZY FORECASTING ADVANTAGE"
    )
    print("=" * 80)

    print(
        advantage_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Training-data scaling mean:",
        scale_mean
    )

    print(
        "Training-data scaling std:",
        scale_std
    )

    results_df.to_csv(
        "fac_forecasting_results.csv",
        index=False
    )

    history_df.to_csv(
        "fac_forecasting_history.csv",
        index=False
    )

    summary_df.to_csv(
        "fac_forecasting_summary.csv",
        index=False
    )

    advantage_df.to_csv(
        "fac_forecasting_advantage.csv",
        index=False
    )

    print()
    print("Saved:")
    print("  fac_forecasting_results.csv")
    print("  fac_forecasting_history.csv")
    print("  fac_forecasting_summary.csv")
    print("  fac_forecasting_advantage.csv")
