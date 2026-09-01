
# ============================================================
# FAC-RNN FORECASTING — CONTEXT ABLATION v1
#
# Goal:
#   Isolate whether fuzzy reasoning itself contributes beyond
#   ordinary learned adaptive alpha.
#
# Controllers:
#
#   A) Fixed-alpha:
#        alpha_t = constant
#
#   B) Value-only adaptive:
#        alpha_t = NN(y_t)
#
#   C) Value+delta adaptive:
#        alpha_t = NN(y_t, delta_t)
#
#   D) Fuzzy value+delta:
#        alpha_t = Fuzzy(y_t, delta_t)
#
# All models share the same recurrent state equation:
#
#   h_t = (1-alpha_t)h_(t-1)
#         + alpha_t tanh(W_h h_(t-1) + W_x x_t + b)
#
# Stability:
#
#   ||W_h||_2 <= ||W_h||_F <= KAPPA
#   alpha_t >= ALPHA_MIN
#
# Therefore:
#
#   L_t <= 1-alpha_t + alpha_t*KAPPA
#       <= 1-ALPHA_MIN*(1-KAPPA) < 1
#
# Benchmark:
#   Same synthetic switching nonlinear forecasting process used
#   in forecasting v1/v2.
#
# Chronological split:
#   60% train / 20% validation / 20% test.
#
# Seeds:
#   42, 123, 456.
#
# Main outputs:
#   - Test MSE/MAE/RMSE
#   - mean/std across seeds
#   - pairwise gains against fixed and learned controllers
#
# This is an attribution experiment, not a final benchmark.
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
                0.10
                *
                y[t - 1] ** 3
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

    return np.stack(
        [ys, delta],
        axis=-1
    ).astype(
        np.float32
    )


def build_windows(
    features,
    target,
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

    return (
        np.asarray(
            X,
            dtype=np.float32
        ),
        np.asarray(
            Y,
            dtype=np.float32
        ).reshape(
            -1,
            1
        )
    )


def prepare_data():

    y, regimes = (
        generate_series()
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
        0,
        train_end,
        LOOKBACK,
        TRAIN_WINDOWS
    )

    va = build_windows(
        features,
        target,
        train_end,
        val_end,
        LOOKBACK,
        VAL_WINDOWS
    )

    te = build_windows(
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
        "Windows:",
        tr[0].shape,
        va[0].shape,
        te[0].shape
    )

    return (
        tr,
        va,
        te,
        mean,
        std
    )


def loader_from_tuple(
    d,
    shuffle
):
    X, Y = d

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
# Contractive recurrent matrix
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


# ============================================================
# Controllers
# ============================================================

class FixedController(nn.Module):

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


class ValueOnlyController(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                1,
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
            x[:, :1]
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


class ValueDeltaController(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                2,
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


class FuzzyController(nn.Module):
    """
    Four-stage transparent fuzzy controller:

      1) neural score maps [value, delta] to a scalar z
      2) Gaussian fuzzy memberships around low/mid/high contexts
      3) normalized rule firing
      4) bounded Sugeno consequent

    Note:
      This controller is intentionally matched to the
      value+delta non-fuzzy controller on the input information.
    """

    def __init__(self):

        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(
                2,
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
            [0.02, 0.10, 0.90]
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
# Common model
# ============================================================

class ContextFACRNN(nn.Module):

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
                FixedController()
            )

        elif controller_type == "value":

            self.controller = (
                ValueOnlyController()
            )

        elif controller_type == "valuedelta":

            self.controller = (
                ValueDeltaController()
            )

        elif controller_type == "fuzzy":

            self.controller = (
                FuzzyController()
            )

        else:

            raise ValueError(
                "Unknown controller type"
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
# Evaluation
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
            pred
            -
            y
        ).pow(2).sum().item()

        ab += (
            pred
            -
            y
        ).abs().sum().item()

        n += y.numel()

    mse = sq / n
    mae = ab / n

    return mse, mae, math.sqrt(mse)


# ============================================================
# Training
# ============================================================

def train_one(
    seed,
    controller_type,
    name,
    train_loader,
    val_loader
):

    seed_all(seed)

    model = (
        ContextFACRNN(
            controller_type
        )
        .to(DEVICE)
    )

    params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
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

    hist = []

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        total = 0.0
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

            total += (
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
            total
            /
            max(
                seen,
                1
            )
        )

        val_mse, val_mae, _ = evaluate(
            model,
            val_loader
        )

        print(
            f"Seed={seed} | {name} | "
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train={train_mse:.6f} | "
            f"val={val_mse:.6f} | "
            f"time={elapsed:.2f}s"
        )

        hist.append({
            "Seed": seed,
            "Architecture": name,
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "epoch_time_sec": elapsed
        })

        if val_mse < best_val - 1e-8:

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
        params,
        best_val,
        best_epoch,
        hist
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print(
        "FAC-RNN FORECASTING — CONTEXT ABLATION v1"
    )
    print("=" * 80)

    print(
        "Device:",
        DEVICE
    )

    print(
        "Controllers:",
        [
            "fixed",
            "value",
            "valuedelta",
            "fuzzy"
        ]
    )

    print(
        "Seeds:",
        SEEDS
    )

    (
        train_data,
        val_data,
        test_data,
        scale_mean,
        scale_std
    ) = prepare_data()

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

    experiments = [
        (
            "fixed",
            "A: Fixed-alpha"
        ),
        (
            "value",
            "B: Learned Value-only alpha"
        ),
        (
            "valuedelta",
            "C: Learned Value+Delta alpha"
        ),
        (
            "fuzzy",
            "D: Fuzzy Value+Delta alpha"
        )
    ]

    results = []
    history = []

    for seed in SEEDS:

        print()
        print(
            "#" * 80
        )
        print(
            f"START SEED {seed}"
        )
        print(
            "#" * 80
        )

        for controller_type, name in experiments:

            model, params, best_val, best_epoch, hist = (
                train_one(
                    seed,
                    controller_type,
                    name,
                    train_loader,
                    val_loader
                )
            )

            history.extend(
                hist
            )

            test_mse, test_mae, test_rmse = (
                evaluate(
                    model,
                    test_loader
                )
            )

            row = {
                "Seed": seed,
                "Architecture": name,
                "Controller": controller_type,
                "Parameters": params,
                "BestEpoch": best_epoch,
                "BestValMSE": best_val,
                "TestMSE": test_mse,
                "TestMAE": test_mae,
                "TestRMSE": test_rmse,
                "MeanEpochTimeSec": float(
                    np.mean([
                        h["epoch_time_sec"]
                        for h in hist
                    ])
                ),
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
            }

            if controller_type == "fuzzy":

                with torch.no_grad():

                    rr = (
                        model.controller
                        .rule_values()
                        .detach()
                        .cpu()
                        .numpy()
                    )

                for i, value in enumerate(
                    rr,
                    1
                ):

                    row[
                        f"AlphaRule{i}"
                    ] = float(value)

                print(
                    "Fuzzy rules:",
                    [
                        float(v)
                        for v in rr
                    ]
                )

            results.append(
                row
            )

            print(
                f"FINAL Seed={seed} | "
                f"{name} | "
                f"TestMSE={test_mse:.8f} | "
                f"TestMAE={test_mae:.8f} | "
                f"RMSE={test_rmse:.8f}"
            )

            # Incremental save.
            pd.DataFrame(
                results
            ).to_csv(
                "fac_context_ablation_results.csv",
                index=False
            )

            pd.DataFrame(
                history
            ).to_csv(
                "fac_context_ablation_history.csv",
                index=False
            )

    results_df = pd.DataFrame(
        results
    )

    history_df = pd.DataFrame(
        history
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
            Parameters=("Parameters", "first"),
            TestMSEMean=(
                "TestMSE",
                "mean"
            ),
            TestMSEStd=(
                "TestMSE",
                "std"
            ),
            TestMAEMean=(
                "TestMAE",
                "mean"
            ),
            TestMAEStd=(
                "TestMAE",
                "std"
            ),
            TestRMSEMean=(
                "TestRMSE",
                "mean"
            ),
            TestRMSEStd=(
                "TestRMSE",
                "std"
            ),
            BestValMSEMean=(
                "BestValMSE",
                "mean"
            ),
            BestValMSEStd=(
                "BestValMSE",
                "std"
            ),
            MeanEpochTimeSec=(
                "MeanEpochTimeSec",
                "mean"
            )
        )
    )

    # Pairwise MSE comparison.
    pivot = results_df.pivot(
        index="Seed",
        columns="Controller",
        values="TestMSE"
    )

    if (
        "fuzzy" in pivot.columns
        and
        "valuedelta" in pivot.columns
    ):

        pivot[
            "Fuzzy_MSE_Reduction_vs_ValueDelta_pct"
        ] = (
            100.0
            *
            (
                pivot["valuedelta"]
                -
                pivot["fuzzy"]
            )
            /
            pivot["valuedelta"]
        )

    if (
        "fuzzy" in pivot.columns
        and
        "value" in pivot.columns
    ):

        pivot[
            "Fuzzy_MSE_Reduction_vs_ValueOnly_pct"
        ] = (
            100.0
            *
            (
                pivot["value"]
                -
                pivot["fuzzy"]
            )
            /
            pivot["value"]
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
        "FINAL CONTEXT ABLATION SUMMARY"
    )
    print("=" * 80)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 80
    )
    print(
        "PER-SEED CONTROLLER ADVANTAGE"
    )
    print(
        "=" * 80
    )

    print(
        advantage_df.to_string(
            index=False
        )
    )

    results_df.to_csv(
        "fac_context_ablation_results.csv",
        index=False
    )

    history_df.to_csv(
        "fac_context_ablation_history.csv",
        index=False
    )

    summary_df.to_csv(
        "fac_context_ablation_summary.csv",
        index=False
    )

    advantage_df.to_csv(
        "fac_context_ablation_advantage.csv",
        index=False
    )

    print()
    print(
        "Saved:"
    )

    print(
        "  fac_context_ablation_results.csv"
    )

    print(
        "  fac_context_ablation_history.csv"
    )

    print(
        "  fac_context_ablation_summary.csv"
    )

    print(
        "  fac_context_ablation_advantage.csv"
    )

    print(
        "Scaling mean:",
        scale_mean
    )

    print(
        "Scaling std:",
        scale_std
    )
