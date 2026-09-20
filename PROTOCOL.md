# Prespecified analysis protocol (written before any policy training or evaluation)

Scope (narrowed): U.S. hospital staff registered nurses (RNs). Physicians are out of scope because
the public calibration data (NSI turnover, NSSRN burnout-at-exit, VHA RN burnout) are RN-specific.

Comparison policies
- Behavioral (unguided): calibrated descriptive model of observed RN exit behavior.
- Expert-guided (rule-based, thresholds fixed here):
  1. Consider exit (move to another healthcare organization) when sustained burnout
     (observed burnout index >= 50 for >= 6 consecutive months) AND poor fit (fit composite < 0.50).
  2. Else renegotiate when compensation < 40th percentile of peer benchmark AND fit is high (>= 0.65)
     (subject to the 12-month renegotiation cooldown).
  3. Else stay (burnout below threshold and/or fit acceptable).
- DRS: discrete Soft Actor-Critic policy; greedy action over feasible (masked) actions at evaluation.

Primary endpoint: Career Value Index (CVI) = 100 x mean monthly utility over 120 months, baseline weights
(fit .30, burnout .25, compensation .20, career capital .15, work-life .10) incl. the sustained-burnout
penalty and job-switching cost. Secondary: Year-10 burnout prevalence, months in burnout, Year-5
compensation percentile, Year-5 career capital, mean fit, job changes, employer turnover cost.

Evaluation: 10,000 held-out hospital RNs (pool seed disjoint from calibration and training), common random
numbers across policies (paired comparisons). Training seeds 1-5 (seed 0 = development only).

Robustness / revision criteria (the model is revised or narrowed if ANY holds):
- C1 Seeds: DRS - comparator CVI difference is <= 0 in any of the 5 independent training seeds, or the
  95% t-interval across seeds includes 0.
- C2 Reward weights: under any prespecified alternative weight vector (equal; wellbeing-priority;
  compensation-priority; career-priority; 3 retrained seeds each), the DRS advantage (evaluated under that
  vector's utility) is <= 0 in any seed.
- C3 Subgroups (early-career <=5 y; mid/late-career >5 y; high baseline burnout; thin labor market):
  DRS is materially worse than a comparator = CVI difference < -1.0 point with 95% CI below 0, or Year-10
  burnout prevalence higher by > 3 percentage points with 95% CI above 0.
- C4 Calibration: any calibration target outside its tolerance on an independent (held-out) population.
  Validation targets (not used in fitting) are reported against the same tolerance rule.
