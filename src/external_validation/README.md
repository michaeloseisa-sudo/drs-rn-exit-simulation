# External validation against the 2022 NSSRN public-use file

1. Download the 2022 NSSRN PUF (ASCII package) from https://data.hrsa.gov/topics/health-workforce/nursing-workforce-survey-data
   and unzip `nssrn_2022_puf_flat.txt` and `nssrn_2022_puf_flat.do` into this folder.
2. Run `python nssrn_targets.py` -> `nssrn_targets.json`.

Population: RNs employed in a hospital setting on 31 Dec 2021 (pn_empset_comb_puf = 1), excluding travel nurses (pn_travel = 2).
Weights: final weight rkrnwgta; standard errors by Successive Difference Replication with 80 replicate weights,
var = (4/80) * sum_r (theta_r - theta)^2. Check: weighted total of all RNs = 4,349,377.
Items (2022 questionnaire): C4 employer tenure (pn_howlong); C40 left primary position (pn_lftwrk);
D1 reasons for leaving incl. burnout (le_lve_*); D2 continued in nursing (le_wrknurs);
J4 burnout frequency 2019-2021 (cv_burnt_*; weekly+ = "a few times a week" or "every day");
I5 position change 2020->2021 (nh_postn). NSSRN data were NOT used to calibrate the simulator.
