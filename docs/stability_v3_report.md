# FAC-RNN Full-Step Jacobian Stability V3

## Experimental purpose

This experiment evaluates the empirical dynamical behavior of the proposed
FAC-RNN under a full-step Jacobian analysis over a continuous 10,000-step
input stream.

Two models are compared:

- C: Fuzzy adaptive temporal scaling + contraction constraint
- D: Fuzzy adaptive temporal scaling without contraction

## Theoretical condition

The contractive model imposes:

||W_h||_F <= 0.9

and therefore:

||W_h||_2 <= 0.9.

With:

alpha_min = 0.02

the resulting worst-case one-step contraction bound is:

q <= 1 - alpha_min(1 - 0.9) = 0.998.

## Main results

For the contractive FAC-RNN (C), averaged over seeds 42, 123, and 456:

- Test MSE: 0.009638
- Mean Jacobian spectral norm: 0.718859
- 95th percentile: 0.733340
- Maximum Jacobian spectral norm: 0.748476
- Fraction of steps with ||J_t||_2 > 1: 0%
- Total theoretical-bound violations: 0
- Mean ||W_h||_2: 0.738447
- Maximum ||W_h||_2: 0.755584
- Maximum hidden-state norm: 3.347116

The analytic Jacobian was independently cross-checked against automatic
differentiation, with maximum absolute discrepancy:

5.96e-08.

For the unconstrained model (D):

- Test MSE: 0.007315
- Mean Jacobian spectral norm: 1.088406
- 95th percentile: 1.124034
- Maximum Jacobian spectral norm: 1.271514
- Fraction of steps with ||J_t||_2 > 1: 99.8867%
- Mean ||W_h||_2: 1.517867
- Maximum ||W_h||_2: 1.632280

## Interpretation

The unconstrained model achieves lower predictive error on this benchmark,
but it loses the contractive dynamical property.

The contractively constrained FAC-RNN sacrifices some predictive flexibility
while maintaining a bounded recurrent operator and empirically non-expansive
one-step state dynamics throughout the full 10,000-step evaluation stream.

These experiments support the dynamical-stability claim; they do not replace
the mathematical proof of the contraction guarantee.

## Reproducibility

Seeds:
42, 123, 456

Stream length:
10,000

The full-step Jacobian is evaluated analytically at every time step and is
cross-checked against automatic differentiation at selected steps.

