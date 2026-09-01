# FAC-RNN Full-Step Jacobian Stability V4
#
# Protocol-matched stability experiment.
#
# IMPORTANT:
#   This version follows the established FAC-RNN FORECASTING v1 protocol:
#   - TOTAL_POINTS = 24000
#   - chronological 60/20/20 split
#   - training-only scaling
#   - lookback = 64
#   - input = [y_t, delta_t]
#   - regime length = 800
#   - seeds = [42, 123, 456]
#   - original fuzzy controller structure, including learned sigma
#
# Models:
#   C = Fuzzy Adaptive alpha + contraction
#   D = Fuzzy Adaptive alpha + NO contraction
#
# Purpose:
#   Re-run the full-step Jacobian stability analysis under the exact
#   established forecasting data/model protocol, rather than the
#   self-contained generator used by V3.
#
# The analytic one-step Jacobian for fixed observable inputs is:
#
#   J_t = (1-alpha_t) I
#         + alpha_t diag(1-tanh(preact_t)^2) W_h
#
# For the contractive model:
#
#   ||J_t||_2 <= 1-alpha_t(1-kappa)
#              <= 1-alpha_min(1-kappa)
#              = 0.998
#
# Autograd cross-checks are included.

import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Configuration copied from the established forecasting v1
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

STREAM_LEN = 10000
AUTOGRAD_CHECKS = 24
AUTOGRAD_TOL = 1e-5

OUT_DIR = Path(
    "fac_longhorizon_full_jacobian_v4_results"
)
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

PIN_MEMORY = (
    DEVICE.type == "cuda"
)

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
# EXACT established forecasting generator
# ============================================================
def generate_series(
    total_points=TOTAL_POINTS,
    seed=2026,
):
    """
    Same deterministic nonlinear switching generator used in the established
    FAC-RNN forecasting v1 protocol.
    """
    rng = np.random.default_rng(seed)

    y = np.zeros(
        total_points,
        dtype=np.float64,
    )

    y[0] = 0.1
    y[1] = 0.12

    regime_len = 800

    for t in range(
        2,
        total_points,
    ):
        regime = (
            (t // regime_len) % 3
        )

        noise = (
            NOISE_STD
            * rng.standard_normal()
        )

        if regime == 0:
            y[t] = (
                0.94 * y[t - 1]
                - 0.08 * y[t - 2]
                + 0.10
                * np.sin(
                    0.055 * t
                    + 0.7 * y[t - 1]
                )
                + noise
            )

        elif regime == 1:
            y[t] = (
                0.72 * y[t - 1]
                + 0.12 * y[t - 2]
                + 0.22
                * np.sin(
                    0.16 * t
                    + 1.2 * y[t - 1]
                )
                + noise
            )

        else:
            y[t] = (
                0.62 * y[t - 1]
                + 0.08 * y[t - 2]
                - 0.10 * y[t - 1] ** 3
                + 0.14
                * np.sin(
                    0.095 * t
                    + 0.8 * y[t - 1]
                )
                + noise
            )

    return y.astype(
        np.float32
    )


# ============================================================
# EXACT established feature construction
# ============================================================
def make_features(
    y,
    mean,
    std,
):
    y_scaled = (
        y - mean
    ) / max(
        std,
        1e-8,
    )

    delta = np.zeros_like(
        y_scaled
    )

    delta[1:] = (
        y_scaled[1:]
        - y_scaled[:-1]
    )

    return np.stack(
        [
            y_scaled,
            delta,
        ],
        axis=-1,
    ).astype(
        np.float32
    )


def build_windows(
    feature_array,
    target_array,
    start,
    end,
    lookback,
    max_windows,
):
    valid_start = (
        start + lookback
    )

    valid_end = min(
        end,
        valid_start + max_windows,
    )

    X = []
    Y = []

    for t in range(
        valid_start,
        valid_end,
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
        np.asarray(
            X,
            dtype=np.float32,
        ),
        np.asarray(
            Y,
            dtype=np.float32,
        ).reshape(
            -1,
            1,
        ),
    )


def prepare_data():
    # The established experiments use one deterministic series generated
    # with seed=2026; training seeds affect model initialization/training.
    y = generate_series(
        TOTAL_POINTS,
        seed=2026,
    )

    train_end = int(
        TOTAL_POINTS
        * TRAIN_FRAC
    )

    val_end = int(
        TOTAL_POINTS
        * (
            TRAIN_FRAC
            + VAL_FRAC
        )
    )

    train_y = y[:train_end]

    mean = float(
        train_y.mean()
    )

    std = float(
        train_y.std()
    )

    features = make_features(
        y,
        mean,
        std,
    )

    target = (
        (y - mean)
        / max(std, 1e-8)
    ).astype(
        np.float32
    )

    Xtr, Ytr = build_windows(
        features,
        target,
        0,
        train_end,
        LOOKBACK,
        TRAIN_WINDOWS,
    )

    Xva, Yva = build_windows(
        features,
        target,
        train_end,
        val_end,
        LOOKBACK,
        VAL_WINDOWS,
    )

    Xte, Yte = build_windows(
        features,
        target,
        val_end,
        TOTAL_POINTS,
        LOOKBACK,
        TEST_WINDOWS,
    )

    print(
        "Chronological split:",
        f"train=[0,{train_end})",
        f"val=[{train_end},{val_end})",
        f"test=[{val_end},{TOTAL_POINTS})",
    )

    print(
        "Window shapes:",
        Xtr.shape,
        Xva.shape,
        Xte.shape,
    )

    print(
        "Training scale:",
        f"mean={mean:.8f}",
        f"std={std:.8f}",
    )

    return (
        Xtr,
        Ytr,
        Xva,
        Yva,
        Xte,
        Yte,
        mean,
        std,
    )


def make_loader(
    X,
    Y,
    shuffle,
):
    return DataLoader(
        TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(Y),
        ),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        pin_memory=PIN_MEMORY,
        num_workers=0,
    )


# ============================================================
# Contractive matrix — same parameterization
# ============================================================
class ContractiveMatrix(nn.Module):
    def __init__(
        self,
        hidden,
    ):
        super().__init__()

        self.raw = nn.Parameter(
            torch.empty(
                hidden,
                hidden,
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
            max=1.0,
        )

        return (
            scale
            * self.raw
        )


# ============================================================
# Original fuzzy controller
# ============================================================
class OriginalFuzzyAlphaController(nn.Module):
    """
    Same controller structure from the established forecasting v1:

      input = [y_t, delta_t]
      score network: 2 -> 6 -> 1
      z = tanh(score)
      Gaussian membership around [-1,0,1]
      learned sigma
      three learned rule consequents
      alpha in [ALPHA_MIN, ALPHA_MAX]
    """

    def __init__(self):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(
                INPUT_DIM,
                6,
            ),
            nn.Tanh(),
            nn.Linear(
                6,
                1,
            ),
        )

        for layer in self.score:
            if isinstance(
                layer,
                nn.Linear,
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
                0.90,
            ],
            dtype=torch.float32,
        )

        p = (
            (
                alpha_init
                - ALPHA_MIN
            )
            / (
                ALPHA_MAX
                - ALPHA_MIN
            )
        ).clamp(
            1e-5,
            1.0 - 1e-5,
        )

        self.rule_logits = nn.Parameter(
            torch.log(
                p
                / (
                    1.0 - p
                )
            )
        )

        self.register_buffer(
            "centers",
            torch.tensor(
                [-1.0, 0.0, 1.0],
                dtype=torch.float32,
            ),
        )

        self.log_sigma = nn.Parameter(
            torch.tensor(
                -0.5,
                dtype=torch.float32,
            )
        )

    def rule_values(self):
        return (
            ALPHA_MIN
            + (
                ALPHA_MAX
                - ALPHA_MIN
            )
            * torch.sigmoid(
                self.rule_logits
            )
        )

    def sigma(self):
        return (
            F.softplus(
                self.log_sigma
            )
            + 0.05
        )

    def forward(
        self,
        x,
    ):
        z = torch.tanh(
            self.score(x).squeeze(-1)
        )

        diff = (
            z.unsqueeze(-1)
            - self.centers.view(
                1,
                3,
            )
        )

        sigma = self.sigma()

        membership = torch.exp(
            -diff.pow(2)
            / (
                2.0
                * sigma.pow(2)
            )
        )

        weights = (
            membership
            / membership.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(
                1e-8
            )
        )

        rules = self.rule_values()

        alpha = (
            weights
            * rules.view(
                1,
                3,
            )
        ).sum(
            dim=-1
        )

        return alpha


# ============================================================
# FAC forecasting model
# ============================================================
class FACForecastRNN(nn.Module):
    def __init__(
        self,
        contraction=True,
    ):
        super().__init__()

        self.use_contraction = (
            contraction
        )

        self.W_h = ContractiveMatrix(
            HIDDEN
        )

        self.W_x = nn.Parameter(
            torch.empty(
                HIDDEN,
                INPUT_DIM,
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
            OriginalFuzzyAlphaController()
        )

        self.readout = nn.Sequential(
            nn.Linear(
                HIDDEN,
                24,
            ),
            nn.Tanh(),
            nn.Linear(
                24,
                1,
            ),
        )

    def recurrent_matrix(self):
        if self.use_contraction:
            return self.W_h()
        return self.W_h.raw

    def step(
        self,
        xt,
        h,
        return_cache=False,
    ):
        W = self.recurrent_matrix()

        input_part = (
            xt @ self.W_x.T
            + self.b
        )

        preact = (
            h @ W.T
            + input_part
        )

        candidate = torch.tanh(
            preact
        )

        alpha = (
            self.controller(
                xt
            ).unsqueeze(
                1
            )
        )

        h_new = (
            (1.0 - alpha)
            * h
            + alpha
            * candidate
        )

        if return_cache:
            return (
                h_new,
                alpha,
                preact,
            )

        return h_new

    def forward(
        self,
        x,
    ):
        B, T, _ = x.shape

        h = torch.zeros(
            B,
            HIDDEN,
            device=x.device,
            dtype=x.dtype,
        )

        for t in range(T):
            h = self.step(
                x[:, t, :],
                h,
            )

        return self.readout(h)

    def fro_norm(self):
        with torch.no_grad():
            return float(
                self.recurrent_matrix().norm(
                    p="fro"
                )
            )

    def spectral_norm(self):
        with torch.no_grad():
            return float(
                torch.linalg.matrix_norm(
                    self.recurrent_matrix(),
                    ord=2,
                )
            )


# ============================================================
# Training
# ============================================================
def evaluate(
    model,
    loader,
):
    model.eval()

    sq = 0.0
    ab = 0.0
    n = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(
                DEVICE,
                non_blocking=PIN_MEMORY,
            )

            y = y.to(
                DEVICE,
                non_blocking=PIN_MEMORY,
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

    mse = sq / max(
        n,
        1,
    )

    mae = ab / max(
        n,
        1,
    )

    return {
        "mse": mse,
        "mae": mae,
        "rmse": math.sqrt(mse),
    }


def run_one(
    seed,
    model_name,
    contraction,
    train_loader,
    val_loader,
    test_loader,
):
    seed_all(seed)

    model = (
        FACForecastRNN(
            contraction=contraction
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
        - ALPHA_MIN
        * (
            1.0
            - KAPPA
        )
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val = float(
        "inf"
    )

    best_state = None
    best_epoch = 0
    bad = 0
    history = []

    for epoch in range(
        1,
        EPOCHS + 1,
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
                non_blocking=PIN_MEMORY,
            )

            y = y.to(
                DEVICE,
                non_blocking=PIN_MEMORY,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            pred = model(x)

            loss = F.mse_loss(
                pred,
                y,
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

            loss_sum += (
                loss.item()
                * x.size(0)
            )

            seen += x.size(0)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - t0
        )

        train_mse = (
            loss_sum
            / max(
                seen,
                1,
            )
        )

        val = evaluate(
            model,
            val_loader,
        )

        print(
            f"[{model_name}] "
            f"seed={seed} "
            f"epoch={epoch:02d} "
            f"train={train_mse:.8f} "
            f"val={val['mse']:.8f} "
            f"time={elapsed:.2f}s"
        )

        history.append(
            dict(
                Seed=seed,
                Model=model_name,
                epoch=epoch,
                train_mse=train_mse,
                val_mse=val["mse"],
                val_mae=val["mae"],
                epoch_time_sec=elapsed,
            )
        )

        if val["mse"] < (
            best_val - 1e-8
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
                f"[{model_name}] "
                f"seed={seed}: early stopping."
            )
            break

    model.load_state_dict(
        best_state
    )

    test = evaluate(
        model,
        test_loader,
    )

    rules = (
        model.controller
        .rule_values()
        .detach()
        .cpu()
        .numpy()
    )

    sigma = float(
        model.controller
        .sigma()
        .detach()
        .cpu()
    )

    result = dict(
        Seed=seed,
        Model=model_name,
        Parameters=params,
        BestEpoch=best_epoch,
        BestValMSE=best_val,
        TestMSE=test["mse"],
        TestMAE=test["mae"],
        TestRMSE=test["rmse"],
        Kappa=KAPPA,
        AlphaMin=ALPHA_MIN,
        AlphaMax=ALPHA_MAX,
        ContractionQBound=q_bound,
        FroNorm=model.fro_norm(),
        SpectralNorm=model.spectral_norm(),
        AlphaRule1=float(rules[0]),
        AlphaRule2=float(rules[1]),
        AlphaRule3=float(rules[2]),
        Sigma=sigma,
        MeanEpochTimeSec=float(
            np.mean(
                [
                    h["epoch_time_sec"]
                    for h in history
                ]
            )
        ),
    )

    print(
        f"FINAL {model_name} "
        f"seed={seed}: "
        f"MSE={test['mse']:.8f} "
        f"||Wh||2={model.spectral_norm():.6f}"
    )

    return (
        model,
        result,
        history,
    )


# ============================================================
# Full-step Jacobian
# ============================================================
@torch.no_grad()
def full_step_jacobian_scan(
    model,
    stream,
):
    """
    Exact analytic J_t at every step.

    alpha_t depends only on x_t=[y_t, delta_t], not h.
    """
    model.eval()

    values = np.asarray(
        stream,
        dtype=np.float32,
    )

    delta = np.zeros_like(
        values
    )

    delta[1:] = (
        values[1:]
        - values[:-1]
    )

    W = model.recurrent_matrix()

    I = torch.eye(
        HIDDEN,
        device=DEVICE,
        dtype=torch.float32,
    )

    h = torch.zeros(
        1,
        HIDDEN,
        device=DEVICE,
        dtype=torch.float32,
    )

    rows = []

    for t in range(
        len(values)
    ):
        xt = torch.tensor(
            [[
                float(values[t]),
                float(delta[t]),
            ]],
            device=DEVICE,
            dtype=torch.float32,
        )

        (
            h_new,
            alpha,
            preact,
        ) = model.step(
            # model expects the full [y,delta] observable input
            xt,
            h,
            return_cache=True,
        )

        a = float(
            alpha.item()
        )

        td = (
            1.0
            - torch.tanh(
                preact
            ).pow(2)
        ).reshape(
            HIDDEN
        )

        J = (
            (1.0 - a)
            * I
            + a
            * (
                td[:, None]
                * W
            )
        )

        svals = torch.linalg.svdvals(
            J
        )

        spec = float(
            svals[0].item()
        )

        fro = float(
            torch.linalg.matrix_norm(
                J,
                ord="fro",
            ).item()
        )

        bound = (
            1.0
            - a
            * (
                1.0
                - KAPPA
            )
        )

        rows.append(
            dict(
                step=t,
                alpha=a,
                jacobian_spectral_norm=spec,
                jacobian_fro_norm=fro,
                theoretical_bound=bound,
                jacobian_minus_bound=(
                    spec - bound
                ),
                exceeds_one=float(
                    spec > 1.0
                ),
                exceeds_bound=float(
                    spec > (
                        bound
                        + AUTOGRAD_TOL
                    )
                ),
                hidden_norm=float(
                    torch.linalg.vector_norm(
                        h_new
                    ).item()
                ),
            )
        )

        h = h_new

    return pd.DataFrame(
        rows
    )


# ============================================================
# Autograd verification
# ============================================================
def autograd_crosscheck(
    model,
    stream,
    steps,
):
    """
    Independent check of the analytic Jacobian.
    """
    model.eval()

    values = np.asarray(
        stream,
        dtype=np.float32,
    )

    delta = np.zeros_like(
        values
    )

    delta[1:] = (
        values[1:]
        - values[:-1]
    )

    selected = set(
        steps
    )

    h = torch.zeros(
        1,
        HIDDEN,
        device=DEVICE,
        dtype=torch.float32,
    )

    rows = []

    for t in range(
        len(values)
    ):
        xt = torch.tensor(
            [[
                float(values[t]),
                float(delta[t]),
            ]],
            device=DEVICE,
            dtype=torch.float32,
        )

        if t not in selected:
            with torch.no_grad():
                h = model.step(
                    xt,
                    h,
                )
            continue

        # Analytic.
        with torch.no_grad():
            W = model.recurrent_matrix()

            h_ana, alpha, preact = (
                model.step(
                    xt,
                    h,
                    return_cache=True,
                )
            )

            I = torch.eye(
                HIDDEN,
                device=DEVICE,
            )

            td = (
                1.0
                - torch.tanh(
                    preact
                ).pow(2)
            ).reshape(
                HIDDEN
            )

            a = alpha.reshape(
                1
            )[0]

            J_ana = (
                (1.0 - a)
                * I
                + a
                * (
                    td[:, None]
                    * W
                )
            )

        # Autograd.
        h_prev = (
            h.detach()
            .clone()
        )

        h_prev.requires_grad_(True)

        h_auto = model.step(
            xt,
            h_prev,
        )

        J_auto = torch.zeros(
            HIDDEN,
            HIDDEN,
            device=DEVICE,
        )

        for j in range(
            HIDDEN
        ):
            g = torch.autograd.grad(
                h_auto[0, j],
                h_prev,
                retain_graph=True,
            )[0]

            J_auto[j, :] = g[0]

        max_diff = float(
            torch.max(
                torch.abs(
                    J_ana
                    - J_auto
                )
            ).item()
        )

        spec_a = float(
            torch.linalg.svdvals(
                J_ana
            )[0].item()
        )

        spec_g = float(
            torch.linalg.svdvals(
                J_auto
            )[0].item()
        )

        rows.append(
            dict(
                step=t,
                max_abs_jacobian_difference=max_diff,
                analytic_spectral_norm=spec_a,
                autograd_spectral_norm=spec_g,
                spectral_difference=abs(
                    spec_a - spec_g
                ),
            )
        )

        h = h_auto.detach()

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print(
        "FAC-RNN FULL-STEP JACOBIAN "
        "STABILITY V4"
    )
    print(
        "Protocol-matched to established forecasting v1"
    )
    print(
        "Device:",
        DEVICE,
    )
    print("=" * 80)

    (
        Xtr,
        Ytr,
        Xva,
        Yva,
        Xte,
        Yte,
        scale_mean,
        scale_std,
    ) = prepare_data()

    train_loader = make_loader(
        Xtr,
        Ytr,
        True,
    )

    val_loader = make_loader(
        Xva,
        Yva,
        False,
    )

    test_loader = make_loader(
        Xte,
        Yte,
        False,
    )

    results = []
    histories = []
    jacobians = []
    traces = []
    checks = []

    # Use a separate long stream generated with the same data-generating
    # process and the same fixed generator seed convention.
    raw_stream = generate_series(
        STREAM_LEN,
        seed=2026,
    )

    # IMPORTANT: use the SAME training-only scaling as the forecasting model.
    # This keeps the stability stream in the model's actual input coordinate
    # system rather than silently changing the controller's operating range.
    stream = (
        raw_stream - scale_mean
    ) / max(
        scale_std,
        1e-8,
    )
    stream = stream.astype(
        np.float32
    )

    check_steps = np.linspace(
        0,
        STREAM_LEN - 1,
        min(
            AUTOGRAD_CHECKS,
            STREAM_LEN,
        ),
        dtype=int,
    ).tolist()

    for seed in SEEDS:
        for model_name, contraction in [
            (
                "C_fuzzy_contractive",
                True,
            ),
            (
                "D_fuzzy_unconstrained",
                False,
            ),
        ]:
            (
                model,
                result,
                history,
            ) = run_one(
                seed,
                model_name,
                contraction,
                train_loader,
                val_loader,
                test_loader,
            )

            jac = full_step_jacobian_scan(
                model,
                stream,
            )

            check = autograd_crosscheck(
                model,
                stream,
                check_steps,
            )

            jac.insert(
                0,
                "Seed",
                seed,
            )

            jac.insert(
                0,
                "Model",
                model_name,
            )

            check.insert(
                0,
                "Seed",
                seed,
            )

            check.insert(
                0,
                "Model",
                model_name,
            )

            results.append(
                {
                    **result,
                    "JacMean": float(
                        jac[
                            "jacobian_spectral_norm"
                        ].mean()
                    ),
                    "JacP95": float(
                        jac[
                            "jacobian_spectral_norm"
                        ].quantile(
                            0.95
                        )
                    ),
                    "JacMax": float(
                        jac[
                            "jacobian_spectral_norm"
                        ].max()
                    ),
                    "JacFractionGt1": float(
                        jac[
                            "exceeds_one"
                        ].mean()
                    ),
                    "BoundMax": float(
                        jac[
                            "theoretical_bound"
                        ].max()
                    ),
                    "BoundViolationCount": int(
                        jac[
                            "exceeds_bound"
                        ].sum()
                    ),
                    "MaxBoundGap": float(
                        jac[
                            "jacobian_minus_bound"
                        ].max()
                    ),
                    "HiddenNormMean": float(
                        jac[
                            "hidden_norm"
                        ].mean()
                    ),
                    "HiddenNormP95": float(
                        jac[
                            "hidden_norm"
                        ].quantile(
                            0.95
                        )
                    ),
                    "HiddenNormMax": float(
                        jac[
                            "hidden_norm"
                        ].max()
                    ),
                    "AutogradMaxDiff": float(
                        check[
                            "max_abs_jacobian_difference"
                        ].max()
                    ),
                    "AutogradMaxSpectralDiff": float(
                        check[
                            "spectral_difference"
                        ].max()
                    ),
                }
            )

            histories.extend(
                history
            )

            jacobians.append(
                jac
            )

            checks.append(
                check
            )

            print(
                f"STABILITY {model_name} "
                f"seed={seed}: "
                f"Jmean={results[-1]['JacMean']:.6f} "
                f"Jp95={results[-1]['JacP95']:.6f} "
                f"Jmax={results[-1]['JacMax']:.6f} "
                f"J>1={results[-1]['JacFractionGt1']:.6f} "
                f"boundViol="
                f"{results[-1]['BoundViolationCount']} "
                f"AutoDiffErr="
                f"{results[-1]['AutogradMaxDiff']:.3e}"
            )

    results_df = pd.DataFrame(
        results
    )

    history_df = pd.DataFrame(
        histories
    )

    jac_df = pd.concat(
        jacobians,
        ignore_index=True,
    )

    check_df = pd.concat(
        checks,
        ignore_index=True,
    )

    results_df.to_csv(
        OUT_DIR
        / "fac_longhorizon_full_jacobian_v4_results.csv",
        index=False,
    )

    history_df.to_csv(
        OUT_DIR
        / "fac_longhorizon_full_jacobian_v4_history.csv",
        index=False,
    )

    jac_df.to_csv(
        OUT_DIR
        / "fac_longhorizon_full_jacobian_v4_all_steps.csv",
        index=False,
    )

    check_df.to_csv(
        OUT_DIR
        / "fac_longhorizon_full_jacobian_v4_autograd_check.csv",
        index=False,
    )

    summary = (
        results_df
        .groupby(
            "Model"
        )
        .agg(
            TestMSEMean=(
                "TestMSE",
                "mean",
            ),
            TestMSEStd=(
                "TestMSE",
                "std",
            ),
            JacMean=(
                "JacMean",
                "mean",
            ),
            JacP95=(
                "JacP95",
                "mean",
            ),
            JacMax=(
                "JacMax",
                "max",
            ),
            JacFractionGt1=(
                "JacFractionGt1",
                "mean",
            ),
            BoundMax=(
                "BoundMax",
                "max",
            ),
            BoundViolationTotal=(
                "BoundViolationCount",
                "sum",
            ),
            MaxBoundGap=(
                "MaxBoundGap",
                "max",
            ),
            HiddenNormMax=(
                "HiddenNormMax",
                "max",
            ),
            FroMean=(
                "FroNorm",
                "mean",
            ),
            SpectralMean=(
                "SpectralNorm",
                "mean",
            ),
            SpectralMax=(
                "SpectralNorm",
                "max",
            ),
            AutogradMaxDiff=(
                "AutogradMaxDiff",
                "max",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUT_DIR
        / "fac_longhorizon_full_jacobian_v4_summary.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print(
        "FINAL SUMMARY"
    )
    print("=" * 80)
    print(
        summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print(
        "THEORETICAL / IMPLEMENTATION CHECK"
    )
    print("=" * 80)

    print(
        "q_bound = "
        f"1 - {ALPHA_MIN}*(1-{KAPPA}) "
        f"= "
        f"{1 - ALPHA_MIN*(1-KAPPA):.6f}"
    )

    print(
        "C expected: "
        "BoundViolationTotal = 0, "
        "JacFractionGt1 = 0."
    )

    print(
        "Autograd max abs difference = "
        f"{check_df['max_abs_jacobian_difference'].max():.3e}"
    )

    print(
        "\nSaved to:",
        OUT_DIR.resolve(),
    )


if __name__ == "__main__":
    main()
