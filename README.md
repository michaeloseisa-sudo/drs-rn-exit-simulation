# Publicly calibrated simulation of RN employment-exit decisions (DRS)

Code for "Reinforcement Learning for Strategic Employment Exit Decisions in Healthcare:
A Decision Recommendation System Framework" (ASEM 2026 IAC). Python 3.12; runs on one CPU.

## Scope
U.S. hospital staff registered nurses, monthly decisions over 120 months, five actions
(stay, renegotiate, exit to another healthcare organization, exit to another sector, self-employment).
`PROTOCOL.md` holds the prespecified comparison policies, endpoints and revision criteria C1-C4,
written before any policy was trained.

## Files (src/)
| File | Purpose |
|---|---|
| `sim.py` | Simulator: parameters (tagged input / assumption / calibrated), dynamics, observations, reward, behavioral and expert policies |
| `burnin.py` | Staggered-entry population burn-in and calibration statistics |
| `calibrate.py` | Calibration and validation targets with sources and tolerances; Nelder-Mead fit -> `calibrated.json` |
| `validate_cal.py` | Held-out calibration check on 3 x 15,000 RNs -> `calibration_heldout.json` |
| `pools.py` | Disjoint evaluation / training / development populations -> `pools.pkl` |
| `sac.py` | Discrete Soft Actor-Critic in JAX/Optax |
| `dev_tune.py` | Development-seed (seed 0) tuning run; never used for reported results |
| `run_all.py` | Trains 17 policies: baseline weights seeds 1-5; four alternative weight vectors seeds 1-3 |
| `evaluate.py` | Paired evaluation with common random numbers; Career Value Index and secondary endpoints |
| `eval_all.py` | Evaluates all policies and exploratory structural perturbations -> `eval_results.pkl` |
| `analyze.py` | Criteria C1-C4, subgroups, early-career table, figures -> `results.json`, `fig_*.png` |

## Reproduce (run from src/)
```
pip install -r ../requirements.txt
python calibrate.py        # ~10 min
python validate_cal.py
python pools.py
python dev_tune.py 40000 0.05   # optional: development seed only
python run_all.py          # 17 runs x ~4.5 min
python eval_all.py
python analyze.py
```
`results/` and `figures/` contain the outputs reported in the paper.
Random seeds: calibration 101; held-out check 202-204; evaluation pool 5001; training pool 7001;
development pool 6001; evaluation noise 9001; training seeds 1-5 (development seed 0).

## Data sources used for calibration and inputs
NSI Nursing Solutions (2026); Mohr et al. (2025) JAMA Netw Open; Shah et al. (2021) JAMA Netw Open;
Fusilier, Stelson & Dill (2026) Health Affairs Scholar; BLS OEWS May 2024; Federal Reserve Bank of
Atlanta Wage Growth Tracker (July 2026). Only published aggregate statistics are used; no individual data.

## Limitations
Simulation only. Parameters marked ASSUMPTION in `sim.py` have no public source; the self-employment
and other-sector paths are weakly identified. External validation is required before any real-world use.

