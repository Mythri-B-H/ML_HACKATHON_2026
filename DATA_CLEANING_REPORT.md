# PathSnap – Data Cleaning & Quality Report

## 1. Quality Audit Summary

A comprehensive automated audit was conducted on all 90 trajectory sets in `data/raw/` (80 training, 10 student testing).

| Check Item | Result | Status | Action Required |
| :--- | :--- | :--- | :--- |
| **Missing Files** | 0 missing files (360/360 present) | PASSED | None |
| **Malformed Lines** | 0 malformed records | PASSED | None |
| **Missing / Null / NaN Values** | 0 missing / NaN values | PASSED | None |
| **Duplicate Timestamps** | 0 duplicate timestamps | PASSED | None |
| **Timestamp Reversals ($t_i < t_{i-1}$)** | 0 timestamp reversals | PASSED | None |
| **Duplicate GPS Points ($x_i == x_{i-1}, y_i == y_{i-1}$)** | 6 points across 227,577 | FLAG | Deduplicate or preserve with zero displacement flag |
| **GPS Speed Jumps ($> 216\text{ km/h} = 60\text{ m/s}$)** | 548 points detected | FLAG | Flag as noisy outliers; apply Kalman/median filtering |
| **Invalid Arc Node References ($u, v \notin V$)** | 0 invalid node references | PASSED | None |
| **Invalid Route Arc References ($a \notin E$)** | 0 invalid arc references | PASSED | None |
| **Route Path Discontinuities ($v_k \neq u_{k+1}$)** | 14 transitions across 7 trajectories | FLAG | Handled via shortest-path graph interpolation in HMM decoder |
| **Temporal Gaps ($\Delta t > 10\text{ s}$)** | 42 gap instances | FLAG | Segment trajectory or relax transition penalty |

---

## 2. Identified Anomalies & Outliers

### A. GPS Noise & Instantaneous Speed Jumps
- **Observed:** 548 observations exhibit apparent speeds exceeding $216\text{ km/h}$ ($60\text{ m/s}$), reaching an extreme of $648.5\text{ km/h}$.
- **Root Cause:** GPS multipath propagation, loss of satellite lock in urban canyons or tunnels, followed by a sudden positional snap.
- **Handling Strategy:**
  1. Flag anomalous speed points using kinematic thresholding ($v_{max} = 150\text{ km/h}$ for passenger cars).
  2. In candidate generation, compute candidate distance from projected position or smoothed Kalman state rather than raw noisy coordinates.
  3. Never silently discard records; record an outlier mask in `data/processed/`.

### B. Duplicate Consecutive GPS Observations
- **Observed:** 6 points have identical coordinates $(lon, lat)$ to their immediate predecessor with $\Delta t = 1.0\text{ s}$.
- **Root Cause:** Vehicle stationary at traffic lights / intersections with zero GPS measurement jitter.
- **Handling Strategy:** Retain stationary points with velocity marked as $0.0\text{ m/s}$, maintaining correct temporal synchronization for the HMM sequence.

### C. Route Discontinuities in Ground Truth
- **Observed:** 14 out of 34,723 route transitions have $v_k \neq u_{k+1}$ in the benchmark road graph.
- **Root Cause:** Benchmark network artifact where dual-carriageway crossover or internal intersection connectors were omitted from ground-truth recording.
- **Handling Strategy:** Use shortest-path Dijkstra graph expansion in `route_decoder.py` to fill 1-hop or 2-hop topological gaps, ensuring 100% continuous road geometry reconstruction.

---

## 3. Data Organization & Pipeline Strategy

In compliance with the project guidelines:
- **Raw Data Unaltered:** `data/raw/` remains completely read-only and untouched.
- **Tiered Data Directory:**
  ```
  data/
  ├── raw/               <- Original untouched .track, .nodes, .arcs, .route
  ├── interim/           <- Parsed JSON/Parquet representation with standardized types
  ├── processed/         <- Cleaned trajectories with kinematic features & candidate arcs
  ├── samples/           <- Small, quick-running representative test fixtures (e.g., 2 sample trips)
  └── reports/           <- Automated cleaning & validation JSON reports
  ```
- **Label Leakage Prevention:**
  - Training set (80 trajectories) is split internally into:
    - 64 training trajectories ($80\%$)
    - 16 validation trajectories ($20\%$)
  - Student test set (10 trajectories) is kept separate for independent evaluation.
  - Evaluator test set is never accessed or inspected during training, parameter tuning, or model selection.
