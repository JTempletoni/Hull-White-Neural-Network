# Neural Network Option Pricing

A comparison of neural network approaches to pricing European call options, produced as part of a postgraduate dissertation. Four models are implemented, ranging from a straightforward Black-Scholes surrogate to a market-calibrated network that incorporates real SPY options data.

---

## Overview

Classical option pricing models require solving a PDE or evaluating a closed-form expression at each query point. Neural networks offer a route to amortise that computation: train once on a large synthetic dataset, then price at inference speed. This project explores how much predictive accuracy is gained by enriching the input feature set — moving from the two core Black-Scholes inputs (moneyness and time to expiry) through to a full set of Greeks calibrated on live market data.

---

## Models

### `BS_Model.py` — Black-Scholes surrogate

Generates 100,000 synthetic parameter sets via **Latin Hypercube Sampling** (LHS) over $(S_0, K, T, \sigma, r)$, computes closed-form Black-Scholes call prices, then trains a 5-layer MLP to learn the map

$$(\text{Moneyness}, T) \;\longmapsto\; C_{BS}$$

Includes an OLS regression baseline to verify that the neural network adds value over a linear approximation. Saves the trained model and scaler via `joblib`.

---

### `HW_Model.py` — Hull-White FFT surrogate

Replaces the Black-Scholes closed form with a **Fourier transform pricer**. The Hull-White characteristic function for log-price under constant volatility is

$$\phi_T(u) = \exp\!\left(iu\mu - \tfrac{1}{2}u^2 \sigma^2 T\right), \quad \mu = \ln S_0 + \left(r - \tfrac{\sigma^2}{2}\right)T$$

Call prices are recovered via the Carr-Madan FFT method with the dampened modified transform

$$\tilde{C}(v) = \frac{e^{-rT}\,\phi_T\!\left(v - (\alpha+1)i\right)}{\alpha^2 + \alpha - v^2 + i(2\alpha+1)v}$$

using $\alpha = 1.5$, $\eta = 0.25$, $N = 4096$. The network architecture and feature set are identical to `BS_Model.py`; the intent is to isolate the effect of the pricing model rather than the network.

---

### `HW based on BS and greek sort of accurate.py` — Greeks-augmented network

Adds the four Black-Scholes Greeks $(\Delta, \Gamma, \mathcal{V}, \rho)$ and the closed-form BS price itself as input features, with Hull-White FFT prices as the training target:

$$(\text{Moneyness},\, T,\, C_{BS},\, \Delta,\, \Gamma,\, \mathcal{V},\, \rho) \;\longmapsto\; C_{HW}$$

The hypothesis is that the Greeks encode curvature and sensitivity information that reduces the learning burden on the network when the target is a different pricing model.

---

### `calibrated_NN.py` — Market-calibrated network

The most involved pipeline. Uses real SPY options data to calibrate the volatility dynamics before generating the training set.

**Pipeline:**

1. **Load & clean** SPY options data (CSV), compute time to expiry in trading years, filter zero-price contracts.
2. **Fit two MLPs** on real market data to learn the drift $a(\sigma_t, t)$ and diffusion $b(\sigma_t, t)$ coefficients of a stochastic volatility SDE:
   $$d\sigma_t = a(\sigma_t, t)\,dt + b(\sigma_t, t)\,dW_t$$
3. **Generate synthetic data** by fitting **kernel density estimates** (KDE) to the empirical distributions of $S$, $K$, $T$ and each Greek, then resampling 200,000 synthetic parameter sets.
4. **Price synthetically** using the Hull-White FFT method, with $\sigma_t$ evolved forward using the learned $a$ and $b$ functions at each time step.
5. **Train a pricing network** on the synthetic Greeks and contract features, with the KDE-calibrated HW prices as targets.

> **Data requirement:** This script reads from a local SPY options CSV. Update the `file_path` variable at the top of the file to point to your copy. The zipped data file is included in the repository.

---

## Repository Structure

| File | Description |
|---|---|
| `BS_Model.py` | Black-Scholes surrogate; LHS synthetic data; MLP on $(T, M)$ |
| `HW_Model.py` | Hull-White FFT surrogate; LHS synthetic data; MLP on $(T, M)$ |
| `HW based on BS and greek sort of accurate.py` | HW target with BS price + Greeks as inputs |
| `calibrated_NN.py` | Real-data calibration via KDE; stochastic-vol drift/diffusion learned from market |

---

## Network Architecture

All models use the same MLP configuration:

```
Input → [256] → [128] → [128] → [64] → [32] → Output
Activation: ReLU
Solver: Adam  |  lr: 0.001  |  batch: 64  |  α (L2): 0.0001
```

Features are standardised with `sklearn.preprocessing.StandardScaler` fit on the training split. An 80/20 train-test split is used throughout, with MSE as the evaluation metric.

---

## Mathematical Motivation

### Universal Approximation and Option Pricing

The theoretical justification for replacing a closed-form pricer with a neural network rests on the **Universal Approximation Theorem** (Cybenko, 1989; Hornik et al., 1991): a feedforward network with at least one hidden layer and a non-polynomial activation function can approximate any continuous function $f: \mathbb{R}^n \to \mathbb{R}$ to arbitrary precision on a compact domain. Formally, for any $\varepsilon > 0$ there exists a network $\hat{f}$ with parameters $\theta$ such that

$$\sup_{x \in K} |f(x) - \hat{f}(x;\theta)| < \varepsilon$$

for compact $K \subset \mathbb{R}^n$. Since option pricing maps — Black-Scholes, Hull-White, and their Greeks — are continuous on compact parameter domains, the theorem guarantees that a sufficiently large network can represent them exactly. The practical question is whether a network of fixed finite depth and width, trained on finite data, is an efficient approximator relative to simply evaluating the formula. For the FFT-based Hull-White pricer, where each evaluation requires a full numerical integration, the answer is yes: the network amortises the computational cost at training time and prices at negligible inference cost thereafter.

### Hull-White Stochastic Volatility

The Hull-White stochastic volatility model specifies that the underlying follows geometric Brownian motion with a volatility process $\sigma_t$ driven by its own SDE:

$$dS_t = rS_t\,dt + \sigma_t S_t\,dW_t^{(1)}$$
$$d\sigma_t = a(\sigma_t, t)\,dt + b(\sigma_t, t)\,dW_t^{(2)}$$

where $a$ and $b$ are the drift and diffusion of volatility respectively. In `calibrated_NN.py` these functions are themselves learned from market data rather than specified parametrically — two MLPs are trained on SPY implied volatility changes to estimate $a$ and $b$, which are then used to forward-simulate $\sigma_t$ inside the FFT pricer.

Conditional on the realised variance path $V_T = \int_0^T \sigma_t^2\,dt$, the log-price is normally distributed and the characteristic function of $\ln S_T$ can be expressed in terms of the moment generating function of $V_T$. In the constant-volatility limit this collapses to the Black-Scholes characteristic function used in `HW_Model.py`:

$$\phi_T(u) = \exp\!\left(iu\mu - \tfrac{1}{2}u^2 \sigma^2 T\right), \qquad \mu = \ln S_0 + \left(r - \frac{\sigma^2}{2}\right)T$$

### Carr-Madan FFT Pricing

To recover call prices across a grid of strikes simultaneously, the Carr-Madan method (1999) introduces a dampening factor $e^{\alpha k}$ on the call price as a function of log-strike $k = \ln K$, making it square-integrable. The modified transform is

$$\tilde{C}(v) = \frac{e^{-rT}\,\phi_T\!\left(v - (\alpha+1)i\right)}{\alpha^2 + \alpha - v^2 + i(2\alpha+1)v}$$

and the call price at log-strike $k$ is recovered via

$$C(k) = \frac{e^{-\alpha k}}{\pi} \int_0^\infty e^{-ivk}\,\tilde{C}(v)\,dv$$

Discretising with spacing $\eta$ and applying the FFT gives prices across $N = 4096$ strikes in a single $O(N \log N)$ operation, with Simpson's rule weights applied to reduce quadrature error. Parameters used throughout: $\alpha = 1.5$, $\eta = 0.25$, $N = 4096$.

---

## Known Issues

The `calibrated_NN.py` script contains a known error: both target arrays `y_a_real` and `y_b_real` are assigned `delta_sigma_real`, meaning the drift and diffusion networks are trained on identical targets. In a correct implementation `y_b_real` should reflect the magnitude of volatility innovations (e.g. squared or absolute changes). This does not affect `BS_Model.py`, `HW_Model.py`, or `HW based on BS and greek sort of accurate.py`. The code is preserved as submitted and will not be updated.

---

## Dependencies

```
numpy
pandas
scikit-learn
scipy
matplotlib
seaborn
joblib
pyDOE          # Latin Hypercube Sampling
statsmodels    # OLS baseline in BS_Model.py
```

Install with:
```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn joblib pyDOE statsmodels
```

If using Google Colab:
```python
!pip install pyDOE
```

---

## References

- Carr, P. & Madan, D. (1999). *Option valuation using the fast Fourier transform.* Journal of Computational Finance, 2(4), 61–73.
- Black, F. & Scholes, M. (1973). *The pricing of options and corporate liabilities.* Journal of Political Economy, 81(3), 637–654.
- Hull, J. & White, A. (1987). *The pricing of options on assets with stochastic volatilities.* Journal of Finance, 42(2), 281–300.
- Cybenko, G. (1989). *Approximation by superpositions of a sigmoidal function.* Mathematics of Control, Signals and Systems, 2(4), 303–314.
- Hornik, K., Stinchcombe, M. & White, H. (1989). *Multilayer feedforward networks are universal approximators.* Neural Networks, 2(5), 359–366.
- Hutchinson, J. M., Lo, A. W. & Poggio, T. (1994). *A nonparametric approach to pricing and hedging derivative securities via learning networks.* Journal of Finance, 49(3), 851–889.
- Liu, S., Oosterlee, C. W. & Bohte, S. M. (2019). *Pricing options and computing implied volatilities using neural networks.* Risks, 7(1), 16.
- McKay, M. D., Beckman, R. J. & Conover, W. J. (1979). *A comparison of three methods for selecting values of input variables in the analysis of output from a computer code.* Technometrics, 21(2), 239–245. *(Latin Hypercube Sampling)*
