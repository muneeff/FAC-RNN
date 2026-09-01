
# ============================================================
# FAC-RNN DATA-SCARCITY v1
#
# Finalized FAC-RNN architecture from the current project.
#
# Research question:
#   Does fuzzy adaptive temporal integration become more useful
#   as the amount of training data decreases?
#
# Controllers:
#   A) Fixed-alpha Contractive RNN
#   B) Learned Non-Fuzzy alpha RNN
#   C) Fuzzy Adaptive alpha RNN
#
# Training-set sizes:
#   1000, 3000, 5000, 9000 windows
#
# Validation and test sets are FIXED across all conditions.
#
# Seeds:
#   42, 123, 456
#
# IMPORTANT:
#   The complete project already established the backbone and
#   controller design. This experiment changes ONLY the number
#   of training windows.
#
# Stability:
#   ||W_h||_2 <= ||W_h||_F <= 0.90
#   alpha >= 0.02
#   q <= 0.998
#
# Outputs:
#   fac_datascarcity_results.csv
#   fac_datascarcity_summary.csv
#   fac_datascarcity_advantage.csv
#   fac_datascarcity_history.csv
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

# Fixed validation/test sets.
VAL_WINDOWS = 2500
TEST_WINDOWS = 2500

TRAIN_SIZES = [
    1000,
    3000,
    5000,
    9000
]

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
# Same synthetic forecasting process
# ============================================================

def generate_series(
    total_points=TOTAL_POINTS,
    seed=2026
):

    rng = np.random.default_rng(
        seed
    )

    y = np.zeros(
        total_points,
        dtype=np.float64
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

        noise = (
            NOISE_STD
            *
            rng.standard_normal()
        )

        if regime == 0:

            y[t] = (
                0.94 * y[t-1]
                -
                0.08 * y[t-2]
                +
                0.10
                *
                np.sin(
                    0.055*t
                    +
                    0.7*y[t-1]
                )
                +
                noise
            )

        elif regime == 1:

            y[t] = (
                0.72 * y[t-1]
                +
                0.12 * y[t-2]
                +
                0.22
                *
                np.sin(
                    0.16*t
                    +
                    1.2*y[t-1]
                )
                +
                noise
            )

        else:

            y[t] = (
                0.62 * y[t-1]
                +
                0.08 * y[t-2]
                -
                0.10*y[t-1]**3
                +
                0.14
                *
                np.sin(
                    0.095*t
                    +
                    0.8*y[t-1]
                )
                +
                noise
            )

    return y.astype(
        np.float32
    )


def make_features(
    y,
    train_end
):

    mean = float(
        y[:train_end].mean()
    )

    std = float(
        y[:train_end].std()
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
        ys[1:] -
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
        ys.astype(np.float32),
        features,
        mean,
        std
    )


def build_windows(
    features,
    target,
    start,
    end,
    max_windows
):

    begin = (
        start + LOOKBACK
    )

    finish = min(
        end,
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


def prepare_dataset():

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

    y = generate_series()

    target, features, mean, std = (
        make_features(
            y,
            train_end
        )
    )

    # Prepare a large chronological training pool.
    Xpool, Ypool = build_windows(
        features,
        target,
        0,
        train_end,
        TRAIN_SIZES[-1]
    )

    # Fixed validation.
    Xval, Yval = build_windows(
        features,
        target,
        train_end,
        val_end,
        VAL_WINDOWS
    )

    # Fixed test.
    Xtest, Ytest = build_windows(
        features,
        target,
        val_end,
        TOTAL_POINTS,
        TEST_WINDOWS
    )

    print(
        "Chronological split:"
    )

    print(
        f"Train pool: {Xpool.shape}"
    )

    print(
        f"Validation: {Xval.shape}"
    )

    print(
        f"Test: {Xtest.shape}"
    )

    return (
        Xpool,
        Ypool,
        Xval,
        Yval,
        Xtest,
        Ytest,
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
                KAPPA/fro,
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

    def forward(
        self,
        x
    ):

        z = self.net(
            x
        ).squeeze(
            -1
        )

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
            1.0-1e-5
        )

        self.rule_logits = nn.Parameter(
            torch.log(
                p/(1-p)
            )
        )

        self.register_buffer(
            "centers",
            torch.tensor(
                [-1.0,0.0,1.0]
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
            ).squeeze(
                -1
            )
        )

        d = (
            z.unsqueeze(-1)
            -
            self.centers.view(
                1,3
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
            ).clamp_min(
                1e-8
            )
        )

        rules = self.rule_values()

        return (
            weights
            *
            rules.view(1,3)
        ).sum(
            dim=-1
        )


# ============================================================
# Common FAC-RNN
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
            torch.zeros(HIDDEN)
        )

        nn.init.xavier_uniform_(
            self.W_x
        )

        if controller_type == "fixed":

            self.controller = (
                FixedController()
            )

        elif controller_type == "nonfuzzy":

            self.controller = (
                NonFuzzyController()
            )

        elif controller_type == "fuzzy":

            self.controller = (
                FuzzyController()
            )

        else:

            raise ValueError(
                controller_type
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

        B,T,_ = x.shape

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

        for t in range(T):

            xt = x[:,t,:]

            alpha = self.controller(
                xt
            ).unsqueeze(1)

            candidate = torch.tanh(
                F.linear(
                    h,
                    wh
                )
                +
                inp[:,t,:]
            )

            h = (
                (1-alpha)*h
                +
                alpha*candidate
            )

        return self.readout(h)


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

        e = p-y

        sq += e.pow(2).sum().item()
        ab += e.abs().sum().item()

        n += y.numel()

    mse = sq / max(n,1)
    mae = ab / max(n,1)

    return (
        mse,
        mae,
        math.sqrt(mse)
    )


# ============================================================
# One run
# ============================================================

def run_one(
    seed,
    train_size,
    controller_type,
    controller_name,
    Xpool,
    Ypool,
    val_loader,
    test_loader
):

    seed_all(seed)

    # Same deterministic subset for the same seed and train size.
    # Shuffle once, then use the first train_size windows.
    rng = np.random.default_rng(
        seed + train_size
    )

    indices = rng.choice(
        len(Xpool),
        size=train_size,
        replace=False
    )

    X = Xpool[
        indices
    ]

    Y = Ypool[
        indices
    ]

    train_loader = make_loader(
        X,
        Y,
        True
    )

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

            optimizer.zero_grad(
                set_to_none=True
            )

            p = model(x)

            loss = F.mse_loss(
                p,
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

        val_mse, val_mae, val_rmse = (
            evaluate(
                model,
                val_loader
            )
        )

        elapsed = (
            time.perf_counter()
            -
            t0
        )

        train_mse = (
            total
            /
            max(seen,1)
        )

        print(
            f"Seed={seed} | N={train_size} | "
            f"{controller_name} | "
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train={train_mse:.6f} | "
            f"val={val_mse:.6f} | "
            f"time={elapsed:.2f}s"
        )

        history.append({
            "Seed": seed,
            "TrainSize": train_size,
            "Architecture": controller_name,
            "Controller": controller_type,
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "epoch_time_sec": elapsed
        })

        if val_mse < (
            best_val - 1e-8
        ):

            best_val = val_mse
            best_epoch = epoch
            bad = 0

            best_state = {
                k: v.detach().cpu().clone()
                for k,v in model.state_dict().items()
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

    test_mse, test_mae, test_rmse = (
        evaluate(
            model,
            test_loader
        )
    )

    mean_epoch = float(
        np.mean([
            h["epoch_time_sec"]
            for h in history
        ])
    )

    result = {
        "Seed": seed,
        "TrainSize": train_size,
        "Architecture": controller_name,
        "Controller": controller_type,
        "Parameters": params,
        "BestEpoch": best_epoch,
        "BestValMSE": best_val,
        "TestMSE": test_mse,
        "TestMAE": test_mae,
        "TestRMSE": test_rmse,
        "MeanEpochTimeSec": mean_epoch,
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
                .cpu()
                .numpy()
            )

        result[
            "AlphaRule1"
        ] = float(rr[0])

        result[
            "AlphaRule2"
        ] = float(rr[1])

        result[
            "AlphaRule3"
        ] = float(rr[2])

    print(
        f"FINAL | Seed={seed} | N={train_size} | "
        f"{controller_name} | "
        f"TestMSE={test_mse:.8f} | "
        f"TestMAE={test_mae:.8f}"
    )

    return (
        result,
        history
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("="*80)
    print(
        "FAC-RNN DATA-SCARCITY v1"
    )
    print("="*80)

    print(
        "Device:",
        DEVICE
    )

    print(
        "Seeds:",
        SEEDS
    )

    print(
        "Train sizes:",
        TRAIN_SIZES
    )

    (
        Xpool,
        Ypool,
        Xval,
        Yval,
        Xtest,
        Ytest,
        scale_mean,
        scale_std
    ) = prepare_dataset()

    val_loader = make_loader(
        Xval,
        Yval,
        False
    )

    test_loader = make_loader(
        Xtest,
        Ytest,
        False
    )

    experiments = [
        (
            "fixed",
            "A: Fixed-alpha"
        ),
        (
            "nonfuzzy",
            "B: Learned Non-Fuzzy alpha"
        ),
        (
            "fuzzy",
            "C: Fuzzy Adaptive alpha"
        )
    ]

    results = []
    histories = []

    for seed in SEEDS:

        print()
        print("#"*80)
        print(
            f"START SEED {seed}"
        )
        print(
            "#"*80
        )

        for train_size in TRAIN_SIZES:

            for controller_type, name in experiments:

                result, history = run_one(
                    seed,
                    train_size,
                    controller_type,
                    name,
                    Xpool,
                    Ypool,
                    val_loader,
                    test_loader
                )

                results.append(result)
                histories.extend(history)

                # Incremental save.
                pd.DataFrame(
                    results
                ).to_csv(
                    "fac_datascarcity_results.csv",
                    index=False
                )

                pd.DataFrame(
                    histories
                ).to_csv(
                    "fac_datascarcity_history.csv",
                    index=False
                )

    results_df = pd.DataFrame(
        results
    )

    history_df = pd.DataFrame(
        histories
    )

    summary_df = (
        results_df
        .groupby(
            [
                "TrainSize",
                "Architecture",
                "Controller"
            ],
            as_index=False
        )
        .agg(
            Seeds=("Seed","count"),
            Parameters=("Parameters","first"),
            TestMSEMean=("TestMSE","mean"),
            TestMSEStd=("TestMSE","std"),
            TestMAEMean=("TestMAE","mean"),
            TestMAEStd=("TestMAE","std"),
            TestRMSEMean=("TestRMSE","mean"),
            TestRMSEStd=("TestRMSE","std"),
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

    # --------------------------------------------------------
    # Fuzzy advantages at each training size.
    # --------------------------------------------------------

    piv = results_df.pivot_table(
        index=[
            "Seed",
            "TrainSize"
        ],
        columns="Controller",
        values="TestMSE"
    ).reset_index()

    if (
        "fuzzy" in piv.columns
        and
        "fixed" in piv.columns
    ):

        piv[
            "Fuzzy_vs_Fixed_Reduction_pct"
        ] = (
            100
            *
            (
                piv["fixed"]
                -
                piv["fuzzy"]
            )
            /
            piv["fixed"]
        )

    if (
        "fuzzy" in piv.columns
        and
        "nonfuzzy" in piv.columns
    ):

        piv[
            "Fuzzy_vs_NonFuzzy_Reduction_pct"
        ] = (
            100
            *
            (
                piv["nonfuzzy"]
                -
                piv["fuzzy"]
            )
            /
            piv["nonfuzzy"]
        )

    # Aggregate advantage by train size.
    advantage_summary = (
        piv
        .groupby(
            "TrainSize",
            as_index=False
        )
        .agg(
            Seeds=("Seed","count"),
            Fuzzy_vs_Fixed_MeanPct=(
                "Fuzzy_vs_Fixed_Reduction_pct",
                "mean"
            ),
            Fuzzy_vs_Fixed_StdPct=(
                "Fuzzy_vs_Fixed_Reduction_pct",
                "std"
            ),
            Fuzzy_vs_NonFuzzy_MeanPct=(
                "Fuzzy_vs_NonFuzzy_Reduction_pct",
                "mean"
            ),
            Fuzzy_vs_NonFuzzy_StdPct=(
                "Fuzzy_vs_NonFuzzy_Reduction_pct",
                "std"
            )
        )
    )

    print()
    print("="*80)
    print(
        "FINAL DATA-SCARCITY SUMMARY"
    )
    print("="*80)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "="*80
    )
    print(
        "FUZZY ADVANTAGE VS TRAINING DATA SIZE"
    )
    print(
        "="*80
    )

    print(
        advantage_summary.to_string(
            index=False
        )
    )

    # Save final.
    results_df.to_csv(
        "fac_datascarcity_results.csv",
        index=False
    )

    history_df.to_csv(
        "fac_datascarcity_history.csv",
        index=False
    )

    summary_df.to_csv(
        "fac_datascarcity_summary.csv",
        index=False
    )

    piv.to_csv(
        "fac_datascarcity_per_seed_advantage.csv",
        index=False
    )

    advantage_summary.to_csv(
        "fac_datascarcity_advantage.csv",
        index=False
    )

    print()
    print(
        "Saved:"
    )

    print(
        "  fac_datascarcity_results.csv"
    )

    print(
        "  fac_datascarcity_summary.csv"
    )

    print(
        "  fac_datascarcity_per_seed_advantage.csv"
    )

    print(
        "  fac_datascarcity_advantage.csv"
    )

    print(
        "  fac_datascarcity_history.csv"
    )

    print(
        "Scaling mean:",
        scale_mean
    )

    print(
        "Scaling std:",
        scale_std
    )
