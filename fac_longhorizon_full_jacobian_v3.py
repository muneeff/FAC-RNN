# FAC-RNN Long-Horizon / Full-Step Jacobian Stability V3
#
# Purpose:
#   Finalize the empirical stability check using the EXACT one-step Jacobian
#   at EVERY step of a 10,000-point continuous stream.
#
# Models:
#   C = Fuzzy adaptive alpha + contraction
#   D = Fuzzy adaptive alpha without contraction
#
# Key mathematical point:
#   alpha_t depends only on (x_t, delta_t), not on h_{t-1}.
#   Therefore, for a fixed input stream:
#
#       h_t = (1-a_t) h_{t-1} + a_t tanh(W_h h_{t-1} + ...)
#
#   has exact Jacobian
#
#       J_t = (1-a_t) I
#             + a_t diag(1 - tanh^2(preact_t)) W_h.
#
#   For the contractive model:
#
#       ||J_t||_2
#       <= (1-a_t) + a_t ||W_h||_2
#       <= 1 - a_t(1-kappa)
#       <= 1 - alpha_min(1-kappa)
#       = 0.998.
#
# This script evaluates the analytic Jacobian at ALL STREAM_LEN steps.
# It also performs a small autograd cross-check to verify the analytic
# Jacobian implementation numerically.
#
# IMPORTANT:
#   Replace make_series() with the EXACT generator used by the established
#   FAC-RNN forecasting experiments before using the results in the paper.

import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIGURATION
# ============================================================
SEEDS = [42, 123, 456]

LOOKBACK = 64
HIDDEN = 48

N_TRAIN = 9000
N_VAL = 2500
N_TEST = 2500

EPOCHS = 20
BATCH_SIZE = 128
LR = 2e-3
WEIGHT_DECAY = 1e-5

STREAM_LEN = 10000

ALPHA_MIN = 0.02
ALPHA_MAX = 0.98
KAPPA = 0.9

INITIAL_RULE_ALPHA = [0.02, 0.10, 0.92]
FUZZY_CENTERS = [-1.0, 0.0, 1.0]
FUZZY_SIGMA = 0.55

# Autograd validation points, separate from the full-step analytic scan.
AUTOGRAD_CHECKS = 24
AUTOGRAD_TOL = 1e-5

OUT_DIR = Path(
    "fac_longhorizon_full_jacobian_v3_results"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA
# ============================================================
def make_series(n, seed):
    """
    Self-contained nonlinear switching series.

    Replace this function with the exact generator from the established
    forecasting experiments before using final paper numbers.
    """
    rng = np.random.default_rng(seed)

    y = np.zeros(n, dtype=np.float32)

    for t in range(2, n):
        r = (t // 1200) % 3

        if r == 0:
            a1, a2, nl, forcing = 0.72, -0.18, 0.10, 0.16
        elif r == 1:
            a1, a2, nl, forcing = 0.42, 0.34, 0.18, 0.11
        else:
            a1, a2, nl, forcing = 0.80, -0.32, 0.07, 0.20

        seasonal = (
            0.08
            * math.sin(2 * math.pi * t / 80.0)
        )

        x = (
            forcing
            * math.sin(2 * math.pi * t / 37.0)
            + seasonal
        )

        y[t] = (
            a1 * y[t - 1]
            + a2 * y[t - 2]
            + nl * math.tanh(y[t - 1])
            + x
            + 0.03 * rng.normal()
        )

    y = (
        y - y.mean()
    ) / (
        y.std() + 1e-8
    )

    return y.astype(np.float32)


def make_windows(series, lookback):
    X, Y = [], []

    for t in range(
        lookback,
        len(series),
    ):
        X.append(
            series[
                t - lookback:t
            ]
        )
        Y.append(
            series[t]
        )

    return (
        np.asarray(
            X,
            dtype=np.float32,
        )[..., None],
        np.asarray(
            Y,
            dtype=np.float32,
        )[..., None],
    )


def stream_features(stream):
    stream = np.asarray(
        stream,
        dtype=np.float32,
    )

    delta = np.zeros_like(stream)

    delta[1:] = (
        stream[1:]
        - stream[:-1]
    )

    return stream, delta


# ============================================================
# MODEL
# ============================================================
class FuzzyAdaptiveRNN(nn.Module):
    def __init__(
        self,
        hidden=48,
        contraction=True,
    ):
        super().__init__()

        self.hidden = hidden
        self.use_contraction = contraction

        self.W_h_raw = nn.Parameter(
            torch.empty(
                hidden,
                hidden,
            )
        )

        self.W_x = nn.Parameter(
            torch.empty(
                hidden,
                1,
            )
        )

        self.b = nn.Parameter(
            torch.zeros(hidden)
        )

        self.readout = nn.Linear(
            hidden,
            1,
        )

        nn.init.orthogonal_(
            self.W_h_raw
        )

        nn.init.xavier_uniform_(
            self.W_x
        )

        # IMPORTANT:
        # alpha depends only on observed (value, delta).
        self.controller = nn.Sequential(
            nn.Linear(2, 24),
            nn.Tanh(),
            nn.Linear(24, 1),
        )

        self.register_buffer(
            "centers",
            torch.tensor(
                FUZZY_CENTERS,
                dtype=torch.float32,
            )
        )

        initial = torch.tensor(
            INITIAL_RULE_ALPHA,
            dtype=torch.float32,
        )

        p = (
            (
                initial - ALPHA_MIN
            )
            / (
                ALPHA_MAX - ALPHA_MIN
            )
        ).clamp(
            1e-5,
            1 - 1e-5,
        )

        self.rule_logits = nn.Parameter(
            torch.log(
                p / (1.0 - p)
            )
        )

    def W_h(self):
        if not self.use_contraction:
            return self.W_h_raw

        fro = torch.linalg.matrix_norm(
            self.W_h_raw,
            ord="fro",
        )

        scale = torch.clamp(
            KAPPA / (
                fro + 1e-12
            ),
            max=1.0,
        )

        return (
            self.W_h_raw
            * scale
        )

    def rule_alphas(self):
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

    def alpha_controller(
        self,
        xt,
        delta_t,
    ):
        controller_input = torch.cat(
            [
                xt,
                delta_t,
            ],
            dim=1,
        )

        z = self.controller(
            controller_input
        )

        d2 = (
            z.unsqueeze(-1)
            - self.centers.view(
                1,
                1,
                -1,
            )
        ) ** 2

        mu = torch.exp(
            -0.5
            * d2
            / (
                FUZZY_SIGMA
                ** 2
            )
        )

        mu = mu / (
            mu.sum(
                dim=-1,
                keepdim=True,
            )
            + 1e-8
        )

        ar = self.rule_alphas()

        alpha = (
            mu
            * ar.view(
                1,
                1,
                -1,
            )
        ).sum(
            dim=-1
        )

        return (
            alpha,
            z,
            mu,
        )

    def step(
        self,
        xt,
        delta_t,
        h,
        return_cache=False,
    ):
        W = self.W_h()

        preact = (
            h @ W.T
            + xt @ self.W_x.T
            + self.b
        )

        proposal = torch.tanh(
            preact
        )

        (
            alpha,
            z,
            mu,
        ) = self.alpha_controller(
            xt,
            delta_t,
        )

        h_new = (
            (
                1.0 - alpha
            )
            * h
            + alpha
            * proposal
        )

        if return_cache:
            return (
                h_new,
                alpha,
                z,
                mu,
                preact,
            )

        return h_new

    def forward(self, x):
        B, T, _ = x.shape

        h = torch.zeros(
            B,
            self.hidden,
            device=x.device,
            dtype=x.dtype,
        )

        previous = torch.zeros_like(
            x[:, 0, :]
        )

        alphas = []

        for t in range(T):
            xt = x[:, t, :]

            delta_t = (
                xt - previous
            )

            h, alpha, _, _, _ = (
                self.step(
                    xt,
                    delta_t,
                    h,
                    return_cache=True,
                )
            )

            alphas.append(
                alpha
            )

            previous = xt

        return (
            self.readout(h),
            torch.cat(
                alphas,
                dim=1,
            ),
        )

    def fro_norm(self):
        with torch.no_grad():
            return float(
                torch.linalg.matrix_norm(
                    self.W_h(),
                    ord="fro",
                )
            )

    def spectral_norm(self):
        with torch.no_grad():
            return float(
                torch.linalg.matrix_norm(
                    self.W_h(),
                    ord=2,
                )
            )


# ============================================================
# TRAINING
# ============================================================
def batches(X, Y, seed, epoch):
    idx = np.arange(
        len(X)
    )

    np.random.default_rng(
        seed + 1009 * epoch
    ).shuffle(idx)

    for s in range(
        0,
        len(idx),
        BATCH_SIZE,
    ):
        j = idx[
            s:s + BATCH_SIZE
        ]

        yield (
            torch.from_numpy(
                X[j]
            ).to(DEVICE),
            torch.from_numpy(
                Y[j]
            ).to(DEVICE),
        )


@torch.no_grad()
def evaluate(
    model,
    X,
    Y,
):
    model.eval()

    predictions = []

    for s in range(
        0,
        len(X),
        BATCH_SIZE,
    ):
        xb = torch.from_numpy(
            X[
                s:s + BATCH_SIZE
            ]
        ).to(DEVICE)

        pred, _ = model(xb)

        predictions.append(
            pred.cpu().numpy()
        )

    pred = np.concatenate(
        predictions
    )

    mse = float(
        np.mean(
            (pred - Y) ** 2
        )
    )

    mae = float(
        np.mean(
            np.abs(
                pred - Y
            )
        )
    )

    return mse, mae


def fit(
    model,
    Xtr,
    Ytr,
    Xv,
    Yv,
    seed,
    name,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val = float("inf")
    best_state = None
    times = []

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        t0 = time.perf_counter()

        for xb, yb in batches(
            Xtr,
            Ytr,
            seed,
            epoch,
        ):
            optimizer.zero_grad(
                set_to_none=True
            )

            pred, _ = model(
                xb
            )

            loss = F.mse_loss(
                pred,
                yb,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

        dt = (
            time.perf_counter()
            - t0
        )

        times.append(dt)

        val_mse, _ = evaluate(
            model,
            Xv,
            Yv,
        )

        if val_mse < best_val:
            best_val = val_mse

            best_state = {
                k: v.detach().cpu().clone()
                for k, v
                in model.state_dict().items()
            }

        print(
            f"[{name}] "
            f"seed={seed} "
            f"epoch={epoch:02d} "
            f"val_mse={val_mse:.8f} "
            f"time={dt:.2f}s"
        )

    model.load_state_dict(
        best_state
    )

    return (
        best_val,
        float(
            np.mean(times)
        ),
    )


# ============================================================
# FULL-STEP ANALYTIC JACOBIAN
# ============================================================
@torch.no_grad()
def full_step_analysis(
    model,
    stream,
):
    """
    Evaluate the exact analytic Jacobian at EVERY stream step.

    For:
        h_new = (1-a)h + a*tanh(preact)
        preact = W h + c
    with a independent of h:

        J = (1-a)I + a*diag(1-tanh(preact)^2) W.
    """
    model.eval()

    values, deltas = (
        stream_features(
            stream
        )
    )

    W = model.W_h()

    I = torch.eye(
        model.hidden,
        device=DEVICE,
        dtype=torch.float32,
    )

    h = torch.zeros(
        1,
        model.hidden,
        device=DEVICE,
        dtype=torch.float32,
    )

    jac_rows = []
    trace_rows = []

    for t in range(
        len(values)
    ):
        xt = torch.tensor(
            [[
                float(values[t])
            ]],
            device=DEVICE,
            dtype=torch.float32,
        )

        dt = torch.tensor(
            [[
                float(deltas[t])
            ]],
            device=DEVICE,
            dtype=torch.float32,
        )

        (
            h_new,
            alpha,
            z,
            mu,
            preact,
        ) = model.step(
            xt,
            dt,
            h,
            return_cache=True,
        )

        a = alpha.reshape(
            1
        )[0]

        # Exact derivative of tanh.
        tanh_derivative = (
            1.0
            - torch.tanh(
                preact
            ).pow(2)
        ).reshape(
            model.hidden
        )

        J = (
            (1.0 - a)
            * I
            + a
            * (
                tanh_derivative[:, None]
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

        # Universal bound under ||W||2 <= kappa.
        theoretical_bound = (
            1.0
            - float(a.item())
            * (
                1.0
                - KAPPA
            )
        )

        jac_rows.append(
            dict(
                step=t,
                alpha=float(
                    a.item()
                ),
                jacobian_spectral_norm=spec,
                jacobian_fro_norm=fro,
                theoretical_bound=theoretical_bound,
                jacobian_minus_bound=(
                    spec
                    - theoretical_bound
                ),
                exceeds_one=float(
                    spec > 1.0
                ),
                exceeds_bound=float(
                    spec
                    > theoretical_bound
                    + AUTOGRAD_TOL
                ),
            )
        )

        trace_rows.append(
            dict(
                step=t,
                hidden_norm=float(
                    torch.linalg.vector_norm(
                        h_new
                    ).item()
                ),
                alpha=float(
                    a.item()
                ),
                z=float(
                    z.item()
                ),
                mu1=float(
                    mu.reshape(-1, 3)[0, 0]
                ),
                mu2=float(
                    mu.reshape(-1, 3)[0, 1]
                ),
                mu3=float(
                    mu.reshape(-1, 3)[0, 2]
                ),
            )
        )

        h = h_new

    return (
        pd.DataFrame(jac_rows),
        pd.DataFrame(trace_rows),
    )


# ============================================================
# AUTOGRAD CROSS-CHECK
# ============================================================
def autograd_crosscheck(
    model,
    stream,
    steps,
):
    """
    Verify the analytic Jacobian against autograd at selected steps.

    A mismatch here invalidates the full-step analytic scan, so this check
    must pass before interpreting the results.
    """
    model.eval()

    values, deltas = (
        stream_features(
            stream
        )
    )

    h = torch.zeros(
        1,
        model.hidden,
        device=DEVICE,
        dtype=torch.float32,
    )

    rows = []

    selected = set(
        steps
    )

    for t in range(
        len(values)
    ):
        xt = torch.tensor(
            [[
                float(values[t])
            ]],
            device=DEVICE,
            dtype=torch.float32,
        )

        dt = torch.tensor(
            [[
                float(deltas[t])
            ]],
            device=DEVICE,
            dtype=torch.float32,
        )

        if t not in selected:
            with torch.no_grad():
                h = model.step(
                    xt,
                    dt,
                    h,
                )
            continue

        # --- analytic ---
        with torch.no_grad():
            h_ana, alpha, _, _, preact = (
                model.step(
                    xt,
                    dt,
                    h,
                    return_cache=True,
                )
            )

            W = model.W_h()

            I = torch.eye(
                model.hidden,
                device=DEVICE,
            )

            td = (
                1.0
                - torch.tanh(
                    preact
                ).pow(2)
            ).reshape(
                model.hidden
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

        # --- autograd ---
        h_prev = h.detach().clone()
        h_prev.requires_grad_(True)

        h_auto = model.step(
            xt,
            dt,
            h_prev,
        )

        J_auto = torch.zeros(
            model.hidden,
            model.hidden,
            device=DEVICE,
        )

        for j in range(
            model.hidden
        ):
            g = torch.autograd.grad(
                h_auto[0, j],
                h_prev,
                retain_graph=True,
            )[0]

            J_auto[j, :] = g[0]

        diff = float(
            torch.max(
                torch.abs(
                    J_ana
                    - J_auto
                )
            ).item()
        )

        spec_ana = float(
            torch.linalg.svdvals(
                J_ana
            )[0].item()
        )

        spec_auto = float(
            torch.linalg.svdvals(
                J_auto
            )[0].item()
        )

        rows.append(
            dict(
                step=t,
                max_abs_jacobian_difference=diff,
                analytic_spectral_norm=spec_ana,
                autograd_spectral_norm=spec_auto,
                spectral_difference=(
                    abs(
                        spec_ana
                        - spec_auto
                    )
                ),
            )
        )

        h = h_auto.detach()

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print(
        "FAC-RNN LONG-HORIZON / "
        "FULL-STEP JACOBIAN STABILITY V3"
    )
    print(
        "Device:",
        DEVICE,
    )
    print("=" * 80)

    total = (
        LOOKBACK
        + N_TRAIN
        + N_VAL
        + N_TEST
    )

    all_results = []
    all_jac = []
    all_trace = []
    all_checks = []

    check_steps = np.linspace(
        0,
        STREAM_LEN - 1,
        AUTOGRAD_CHECKS,
        dtype=int,
    ).tolist()

    for seed in SEEDS:
        print("\n" + "#" * 80)
        print(
            "SEED",
            seed,
        )
        print("#" * 80)

        seed_all(seed)

        series = make_series(
            total,
            seed,
        )

        X, Y = make_windows(
            series,
            LOOKBACK,
        )

        Xtr = X[
            :N_TRAIN
        ]

        Ytr = Y[
            :N_TRAIN
        ]

        Xv = X[
            N_TRAIN:
            N_TRAIN + N_VAL
        ]

        Yv = Y[
            N_TRAIN:
            N_TRAIN + N_VAL
        ]

        s0 = (
            N_TRAIN
            + N_VAL
        )

        Xt = X[
            s0:
            s0 + N_TEST
        ]

        Yt = Y[
            s0:
            s0 + N_TEST
        ]

        for name, contraction in [
            (
                "C_fuzzy_contractive",
                True,
            ),
            (
                "D_fuzzy_unconstrained",
                False,
            ),
        ]:
            seed_all(seed)

            model = FuzzyAdaptiveRNN(
                hidden=HIDDEN,
                contraction=contraction,
            ).to(DEVICE)

            best_val, mean_sec = fit(
                model,
                Xtr,
                Ytr,
                Xv,
                Yv,
                seed,
                name,
            )

            test_mse, test_mae = (
                evaluate(
                    model,
                    Xt,
                    Yt,
                )
            )

            stream = make_series(
                STREAM_LEN,
                seed + 10000,
            )

            jac, trace = (
                full_step_analysis(
                    model,
                    stream,
                )
            )

            check = (
                autograd_crosscheck(
                    model,
                    stream,
                    check_steps,
                )
            )

            # Add IDs.
            jac.insert(
                0,
                "seed",
                seed,
            )

            jac.insert(
                0,
                "model",
                name,
            )

            trace.insert(
                0,
                "seed",
                seed,
            )

            trace.insert(
                0,
                "model",
                name,
            )

            check.insert(
                0,
                "seed",
                seed,
            )

            check.insert(
                0,
                "model",
                name,
            )

            all_jac.append(jac)
            all_trace.append(trace)
            all_checks.append(check)

            rule_values = (
                model.rule_alphas()
                .detach()
                .cpu()
                .numpy()
            )

            row = dict(
                model=name,
                seed=seed,
                best_val_mse=best_val,
                test_mse=test_mse,
                test_mae=test_mae,

                fro_norm=model.fro_norm(),
                spectral_norm=model.spectral_norm(),

                rule1_alpha=float(
                    rule_values[0]
                ),
                rule2_alpha=float(
                    rule_values[1]
                ),
                rule3_alpha=float(
                    rule_values[2]
                ),

                hidden_norm_mean=float(
                    trace[
                        "hidden_norm"
                    ].mean()
                ),
                hidden_norm_p95=float(
                    trace[
                        "hidden_norm"
                    ].quantile(
                        0.95
                    )
                ),
                hidden_norm_max=float(
                    trace[
                        "hidden_norm"
                    ].max()
                ),
                hidden_norm_final=float(
                    trace[
                        "hidden_norm"
                    ].iloc[-1]
                ),

                jac_all_mean=float(
                    jac[
                        "jacobian_spectral_norm"
                    ].mean()
                ),
                jac_all_median=float(
                    jac[
                        "jacobian_spectral_norm"
                    ].median()
                ),
                jac_all_p95=float(
                    jac[
                        "jacobian_spectral_norm"
                    ].quantile(
                        0.95
                    )
                ),
                jac_all_max=float(
                    jac[
                        "jacobian_spectral_norm"
                    ].max()
                ),
                jac_fraction_gt1=float(
                    jac[
                        "exceeds_one"
                    ].mean()
                ),
                bound_max=float(
                    jac[
                        "theoretical_bound"
                    ].max()
                ),
                bound_violation_count=int(
                    jac[
                        "exceeds_bound"
                    ].sum()
                ),
                max_bound_gap=float(
                    jac[
                        "jacobian_minus_bound"
                    ].max()
                ),

                init_gain_placeholder=np.nan,

                autograd_max_abs_diff=float(
                    check[
                        "max_abs_jacobian_difference"
                    ].max()
                ),
                autograd_max_spectral_diff=float(
                    check[
                        "spectral_difference"
                    ].max()
                ),

                mean_epoch_sec=mean_sec,
            )

            all_results.append(
                row
            )

            print(
                f"RESULT {name} "
                f"seed={seed}: "
                f"MSE={test_mse:.8f} "
                f"||Wh||F={model.fro_norm():.6f} "
                f"||Wh||2={model.spectral_norm():.6f} "
                f"Jmean={row['jac_all_mean']:.6f} "
                f"Jp95={row['jac_all_p95']:.6f} "
                f"Jmax={row['jac_all_max']:.6f} "
                f"J>1={row['jac_fraction_gt1']:.4f} "
                f"boundViol={row['bound_violation_count']} "
                f"AutoDiffErr={row['autograd_max_abs_diff']:.2e}"
            )

    results = pd.DataFrame(
        all_results
    )

    jacobian = pd.concat(
        all_jac,
        ignore_index=True,
    )

    trace = pd.concat(
        all_trace,
        ignore_index=True,
    )

    checks = pd.concat(
        all_checks,
        ignore_index=True,
    )

    results.to_csv(
        OUT_DIR
        / "fac_longhorizon_full_jacobian_v3_results.csv",
        index=False,
    )

    jacobian.to_csv(
        OUT_DIR
        / "fac_longhorizon_full_jacobian_v3_all_steps.csv",
        index=False,
    )

    trace.to_csv(
        OUT_DIR
        / "fac_longhorizon_full_jacobian_v3_trace.csv",
        index=False,
    )

    checks.to_csv(
        OUT_DIR
        / "fac_longhorizon_full_jacobian_v3_autograd_check.csv",
        index=False,
    )

    summary = (
        results.groupby(
            "model"
        )
        .agg(
            test_mse_mean=(
                "test_mse",
                "mean",
            ),
            test_mse_std=(
                "test_mse",
                "std",
            ),

            jac_mean=(
                "jac_all_mean",
                "mean",
            ),
            jac_p95=(
                "jac_all_p95",
                "mean",
            ),
            jac_max=(
                "jac_all_max",
                "max",
            ),
            jac_fraction_gt1=(
                "jac_fraction_gt1",
                "mean",
            ),

            bound_max=(
                "bound_max",
                "max",
            ),
            bound_violation_total=(
                "bound_violation_count",
                "sum",
            ),
            max_bound_gap=(
                "max_bound_gap",
                "max",
            ),

            hidden_max=(
                "hidden_norm_max",
                "max",
            ),

            fro_mean=(
                "fro_norm",
                "mean",
            ),
            spectral_mean=(
                "spectral_norm",
                "mean",
            ),
            spectral_max=(
                "spectral_norm",
                "max",
            ),

            autograd_max_diff=(
                "autograd_max_abs_diff",
                "max",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUT_DIR
        / "fac_longhorizon_full_jacobian_v3_summary.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(
        summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("THEORETICAL / IMPLEMENTATION CHECK")
    print("=" * 80)

    q = (
        1.0
        - ALPHA_MIN
        * (
            1.0
            - KAPPA
        )
    )

    print(
        f"q_bound = 1 - "
        f"alpha_min*(1-kappa) = {q:.6f}"
    )

    print(
        "Expected for C:"
        " bound_violation_total = 0"
        " and jac_fraction_gt1 = 0."
    )

    print(
        "Autograd cross-check:"
        f" max absolute Jacobian difference = "
        f"{checks['max_abs_jacobian_difference'].max():.3e}"
    )

    print(
        "\nSaved to:",
        OUT_DIR.resolve(),
    )


if __name__ == "__main__":
    main()
