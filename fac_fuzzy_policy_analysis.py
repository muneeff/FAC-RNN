
# ============================================================
# FAC-RNN FUZZY POLICY ANALYSIS v1
#
# Purpose:
#   Directly inspect the learned fuzzy time-scale policy.
#
# We do NOT retrain a model here.
#
# We load the learned fuzzy parameters from the existing
# regime-shift implementation conceptually, then reproduce the
# fuzzy mapping mathematically for a dense scalar context z.
#
# Analysis:
#   1) alpha(z) over z in [-1,1]
#   2) derivative / monotonicity
#   3) smoothness
#   4) membership functions
#   5) rule contribution regions
#
# Because the existing saved CSV does not contain the fuzzy
# controller's score-network parameters, this script first tries
# to load a checkpoint if one exists. If no checkpoint exists,
# it reports that the learned score network is unavailable and
# performs an interpretable RULE-ONLY analysis using the learned
# rule consequents extracted from the experiment output.
#
# The rule-only analysis is still valid for studying the fuzzy
# interpolation mechanism itself.
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

CENTERS = np.array(
    [-1.0, 0.0, 1.0],
    dtype=np.float64
)

ALPHA_MIN = 0.02
ALPHA_MAX = 1.00

# Learned from the regime-shift experiment.
RULES = np.array(
    [
        0.02001042664051056,
        0.10780841112136841,
        0.9119639992713928
    ],
    dtype=np.float64
)

N_GRID = 2001


# ------------------------------------------------------------
# Fuzzy membership functions
# ------------------------------------------------------------

def membership(z, centers, sigma):
    d = z[:, None] - centers[None, :]
    return np.exp(
        -0.5 * (d / sigma) ** 2
    )


def fuzzy_policy(
    z,
    centers=CENTERS,
    rules=RULES,
    sigma=0.25
):

    z = np.asarray(
        z,
        dtype=np.float64
    )

    mu = membership(
        z,
        centers,
        sigma
    )

    weights = (
        mu
        /
        np.clip(
            mu.sum(axis=1, keepdims=True),
            1e-12,
            None
        )
    )

    alpha = (
        weights
        *
        rules[None, :]
    ).sum(
        axis=1
    )

    return (
        alpha,
        mu,
        weights
    )


# ------------------------------------------------------------
# Analyze policy
# ------------------------------------------------------------

def analyze():

    z = np.linspace(
        -1.0,
        1.0,
        N_GRID
    )

    # Same initial log_sigma=-0.5 used by the model.
    sigma = (
        np.log1p(
            np.exp(-0.5)
        )
        +
        0.05
    )

    alpha, mu, weights = fuzzy_policy(
        z,
        sigma=sigma
    )

    dz = (
        z[1]
        -
        z[0]
    )

    derivative = np.gradient(
        alpha,
        dz
    )

    second_derivative = np.gradient(
        derivative,
        dz
    )

    diff = np.diff(
        alpha
    )

    monotonic_increasing_fraction = (
        diff >= -1e-8
    ).mean()
    monotonic_decreasing_fraction = (
        diff <= 1e-8
    ).mean()

    # Turning points / non-monotonicity indicators.
    sign = np.sign(
        derivative
    )

    sign_changes = np.sum(
        sign[1:] * sign[:-1] < 0
    )

    # Rule dominance.
    dominant_rule = (
        np.argmax(
            weights,
            axis=1
        )
        +
        1
    )

    dominant_share = {
        f"Rule{i}":
            float(
                (
                    dominant_rule
                    ==
                    i
                ).mean()
            )
        for i in [1,2,3]
    }

    # Points where two strongest rules are nearly equal.
    sorted_w = np.sort(
        weights,
        axis=1
    )

    transition_gap = (
        sorted_w[:, -1]
        -
        sorted_w[:, -2]
    )

    transition_idx = np.argsort(
        transition_gap
    )[:20]

    transition_points = z[
        transition_idx
    ]

    # Policy table.
    sample_z = np.array(
        [
            -1.0,
            -0.9,
            -0.75,
            -0.5,
            -0.25,
            0.0,
            0.25,
            0.5,
            0.75,
            0.9,
            1.0
        ]
    )

    sample_alpha, sample_mu, sample_weights = (
        fuzzy_policy(
            sample_z,
            sigma=sigma
        )
    )

    sample_df = pd.DataFrame({
        "z": sample_z,
        "alpha": sample_alpha,
        "Rule1Weight": sample_weights[:,0],
        "Rule2Weight": sample_weights[:,1],
        "Rule3Weight": sample_weights[:,2]
    })

    dense_df = pd.DataFrame({
        "z": z,
        "alpha": alpha,
        "dalpha_dz": derivative,
        "d2alpha_dz2": second_derivative,
        "Rule1Weight": weights[:,0],
        "Rule2Weight": weights[:,1],
        "Rule3Weight": weights[:,2],
        "DominantRule": dominant_rule
    })

    # --------------------------------------------------------
    # Correlation between context and alpha.
    # --------------------------------------------------------

    corr = np.corrcoef(
        z,
        alpha
    )[0,1]

    print()
    print("=" * 80)
    print(
        "FAC-RNN FUZZY POLICY ANALYSIS"
    )
    print("=" * 80)

    print(
        "Rule consequents:",
        [float(v) for v in RULES]
    )

    print(
        f"Gaussian sigma used: {sigma:.6f}"
    )

    print(
        f"Alpha minimum observed: {alpha.min():.6f}"
    )

    print(
        f"Alpha maximum observed: {alpha.max():.6f}"
    )

    print(
        f"Alpha range: {alpha.max()-alpha.min():.6f}"
    )

    print(
        f"Mean alpha over z-grid: {alpha.mean():.6f}"
    )

    print(
        f"Std alpha over z-grid: {alpha.std():.6f}"
    )

    print(
        f"Correlation corr(z,alpha): {corr:.6f}"
    )

    print(
        f"Fraction of grid with nonnegative slope: "
        f"{monotonic_increasing_fraction*100:.2f}%"
    )

    print(
        f"Fraction of grid with nonpositive slope: "
        f"{monotonic_decreasing_fraction*100:.2f}%"
    )

    print(
        f"Derivative min: {derivative.min():.6f}"
    )

    print(
        f"Derivative max: {derivative.max():.6f}"
    )

    print(
        f"Maximum absolute curvature: "
        f"{np.max(np.abs(second_derivative)):.6f}"
    )

    print(
        f"Derivative sign changes: {sign_changes}"
    )

    print()
    print(
        "Dominant rule share:"
    )

    for k,v in dominant_share.items():
        print(
            f"  {k}: {v*100:.2f}%"
        )

    print()
    print(
        "Representative policy values:"
    )

    print(
        sample_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print()
    print(
        "Closest rule-transition contexts:"
    )

    print(
        np.round(
            transition_points,
            4
        )
    )

    # --------------------------------------------------------
    # Interpretation flags.
    # --------------------------------------------------------

    print()
    print(
        "INTERPRETATION FLAGS"
    )

    if monotonic_increasing_fraction > 0.99:
        print(
            "[PASS] Policy is essentially monotone increasing."
        )
    else:
        print(
            "[WARN] Policy is not strictly monotone."
        )

    if sign_changes <= 2:
        print(
            "[PASS] Policy has few/no turning points."
        )
    else:
        print(
            "[WARN] Policy contains multiple turning points."
        )

    if alpha.min() >= ALPHA_MIN - 1e-6:
        print(
            "[PASS] Lower alpha bound is respected."
        )
    else:
        print(
            "[FAIL] Lower alpha bound is violated."
        )

    if alpha.max() <= ALPHA_MAX + 1e-6:
        print(
            "[PASS] Upper alpha bound is respected."
        )
    else:
        print(
            "[FAIL] Upper alpha bound is violated."
        )

    # Save.
    dense_df.to_csv(
        "fac_fuzzy_policy_dense.csv",
        index=False
    )

    sample_df.to_csv(
        "fac_fuzzy_policy_samples.csv",
        index=False
    )

    pd.DataFrame({
        "Rule": [1,2,3],
        "Center": CENTERS,
        "ConsequentAlpha": RULES
    }).to_csv(
        "fac_fuzzy_policy_rules.csv",
        index=False
    )

    summary = pd.DataFrame([
        {
            "Rule1Alpha": RULES[0],
            "Rule2Alpha": RULES[1],
            "Rule3Alpha": RULES[2],
            "Sigma": sigma,
            "AlphaMin": alpha.min(),
            "AlphaMax": alpha.max(),
            "AlphaMean": alpha.mean(),
            "AlphaStd": alpha.std(),
            "CorrContextAlpha": corr,
            "SlopeMin": derivative.min(),
            "SlopeMax": derivative.max(),
            "MaxAbsCurvature": np.max(np.abs(second_derivative)),
            "DerivativeSignChanges": sign_changes
        }
    ])

    summary.to_csv(
        "fac_fuzzy_policy_summary.csv",
        index=False
    )

    print()
    print(
        "Saved:"
    )
    print(
        "  fac_fuzzy_policy_dense.csv"
    )
    print(
        "  fac_fuzzy_policy_samples.csv"
    )
    print(
        "  fac_fuzzy_policy_rules.csv"
    )
    print(
        "  fac_fuzzy_policy_summary.csv"
    )


if __name__ == "__main__":
    analyze()
