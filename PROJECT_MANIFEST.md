\# PROJECT MANIFEST: DynamoSpline-RNN



\## 1. Executive Summary

\- \*\*Title\*\*: Global Dynamical Stability and Bounded Gradient Dynamics in Recurrent Neural Networks via Differentiable Monotonic Fuzzy Splines

\- \*\*Target Journal\*\*: IEEE Transactions on Neural Networks and Learning Systems (TNNLS) - Q1

\- \*\*Primary Objective\*\*: Prove theoretically and empirically that Differentiable Monotonic Fuzzy Activation Units (MFAU / DynamoSpline) bound the Lipschitz constant of RNN Jacobian matrices, preventing exploding/vanishing gradients without needing hardware acceleration.



\## 2. Core Theoretical Framework

\- \*\*Function Formulation\*\*:

&nbsp; $$ \\phi(x) = \\sum\_{k=1}^{K} \\mu\_k(x) y\_k $$

\- \*\*Constraints\*\*:

&nbsp; 1. Monotonicity: $y\_1 \\le y\_2 \\le \\dots \\le y\_K$

&nbsp; 2. Boundedness: $y\_{\\min} \\le y\_k \\le y\_{\\max}$

\- \*\*Gradient Enforcement (PyTorch Parameterization)\*\*:

&nbsp; $$ y\_k = y\_{k-1} + \\text{softplus}(\\delta\_k) $$



\## 3. Theoretical Roadmap (Mathematical Proofs)

\- \*\*Theorem 1\*\*: Lipschitz Continuity and Bounded Derivatives ($0 < L\_{\\min} \\le \\phi'(x) \\le L\_{\\max} < \\infty$).

\- \*\*Theorem 2\*\*: Global Asymptotic Stability of RNN Hidden States (Eigenvalue Containment of Jacobian Matrix $J\_t$).

\- \*\*Theorem 3\*\*: Gradient Bound Preservation over time $T \\to \\infty$ ($\\left\\| \\frac{\\partial L}{\\partial W} \\right\\| \\ge C > 0$).



\## 4. Empirical Roadmap (Python/PyTorch Verification)

\- \*\*Adding Problem (Synthetic Benchmark)\*\*: Test for long time horizons $T \\in \\{100, 500, 1000\\}$.

\- \*\*Gradient Dynamics Tracking\*\*: Plot $\\left\\| \\nabla\_W L \\right\\|$ over timesteps $T$ comparing DynamoSpline vs Vanilla LSTM vs GRU.

\- \*\*sMNIST \& HAR Benchmarks\*\*: Real-world sequential evaluation.



\## 5. Project Execution Instructions for AI Assistant

When starting a new session, read `STATE.md` immediately to determine the current execution phase and resume without setup questions.

