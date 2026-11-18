# PathSnap – Machine Learning & HMM Sequence Reconstruction Documentation

## 1. Overview & Problem Formulation

In GPS route reconstruction, vehicles traverse directed road networks while recording noisy, discrete GPS coordinates at regular temporal intervals (typically 1 Hz).

PathSnap solves the map-matching problem in two integrated stages:
1. **Candidate Classification & Emission Scoring:** A supervised machine learning model predicts whether a candidate road arc is the true arc matching the GPS observation, scoring candidate suitability using geometric and kinematic features.
2. **Global Sequence Optimization:** A Hidden Markov Model (HMM) solved via the Viterbi algorithm identifies the most likely sequence of connected road segments across time, enforcing topological road network continuity.

---

## 2. Feature Engineering

For each candidate road arc $c_{t, i} = (u, v)$ and GPS measurement $z_t = (lon_t, lat_t, t)$, a 9-dimensional feature vector $\mathbf{x} \in \mathbb{R}^9$ is extracted:

| Feature Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `perp_dist` | Float | Meters | Orthogonal distance from GPS point $z_t$ to line segment $(u, v)$. |
| `heading_diff` | Float | Degrees | Absolute angular difference $\|\theta_{GPS} - \theta_{arc}\| \in [0^\circ, 180^\circ]$ between vehicle bearing and road segment orientation. |
| `proj_ratio` | Float | $[0, 1]$ | Normalized scalar projection coordinate $t \in [0, 1]$ of the GPS point along $(u, v)$. |
| `dist_to_u` | Float | Meters | Great-circle Haversine distance to arc source node $u$. |
| `dist_to_v` | Float | Meters | Great-circle Haversine distance to arc target node $v$. |
| `arc_length` | Float | Meters | Total geographic length of the candidate road arc. |
| `spatial_rank` | Integer | $\{0, 1, \dots\}$ | Rank index when sorting candidates by perpendicular distance (0 = nearest). |
| `speed_mps` | Float | m/s | Instantaneous vehicle speed $v_t = \frac{\Delta d}{\Delta t}$. |
| `speed_consistency`| Float | Ratio | Speed-to-length ratio $\frac{v_t}{\text{length} + 10}$ capturing traversal plausibility. |

All features are normalized via `StandardScaler` during model training.

---

## 3. The 5 Machine Learning Models

### Model 1: Logistic Regression (Linear Probabilistic Baseline)
- **Family:** Generalized Linear Model (GLM)
- **Loss Function:** Binary Cross-Entropy (Log Loss) with $\ell_2$ regularization
- **Probability Output:** $P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$
- **Purpose:** High-speed baseline providing well-calibrated sigmoid probability estimates with minimal inference latency ($< 0.1\text{ ms}$).

### Model 2: Random Forest Classifier (Bagging Ensemble)
- **Family:** Decision Tree Ensemble (Bagging)
- **Hyperparameters:** $100$ trees, `max_depth=12`, `min_samples_split=5`, `class_weight='balanced'`
- **Probability Output:** Tree ensemble majority vote: $P(y=1 \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^B P_b(y=1 \mid \mathbf{x})$
- **Purpose:** Robust non-linear interactions between orthogonal distance, heading discrepancy, and candidate rank; handles noisy outliers effectively.

### Model 3: Histogram Gradient Boosting (GBDT)
- **Family:** Gradient Boosted Decision Trees
- **Hyperparameters:** $100$ boosting iterations, `max_depth=8`, `learning_rate=0.1`, `class_weight='balanced'`
- **Probability Output:** Logistic transform of accumulated tree residuals: $P(y=1 \mid \mathbf{x}) = \sigma\left(\sum_{m=1}^M \gamma_m h_m(\mathbf{x})\right)$
- **Purpose:** Optimal tabular ranking accuracy, binning continuous features into integer histograms for fast evaluation.

### Model 4: Multi-Layer Perceptron (MLP Neural Network)
- **Family:** Deep Feedforward Neural Network
- **Architecture:** Input ($9$) $\to$ Dense ($64$, ReLU) $\to$ Dense ($32$, ReLU) $\to$ Output ($1$, Sigmoid)
- **Optimization:** Adam optimizer with early stopping on validation loss
- **Purpose:** Learns smooth non-linear continuous geometric representations of spatial proximity and heading angles.

### Model 5: Support Vector Machine (Kernel Margin Classifier)
- **Family:** Maximal Margin Classifier with Probability Calibration
- **Formulation:** Linear SVC with Platt scaling via 3-fold `CalibratedClassifierCV`
- **Purpose:** Maximizes margin of separation between true road arcs and adjacent parallel streets/cross-lanes.

---

## 4. Empirical Evaluation Results

Evaluated on held-out validation trajectories ($16$ trajectories, $36,901$ candidate-observation pairs) with zero trajectory-level data leakage:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Inference Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | $0.7614$ | $0.6370$ | $0.7669$ | $0.6960$ | $0.8274$ | $1.4\text{ ms}$ |
| **Random Forest** | $0.7786$ | $0.6747$ | $0.7305$ | $0.7015$ | $0.8348$ | $4.2\text{ ms}$ |
| **HistGradientBoosting** | $0.7769$ | $0.6661$ | $0.7487$ | $0.7050$ | $0.8388$ | $2.1\text{ ms}$ |
| **Multi-Layer Perceptron (MLP)** | **0.8108** | **0.7787** | $0.6547$ | **0.7114** | **0.8500** | $2.8\text{ ms}$ |
| **Support Vector Machine (SVM)** | $0.7860$ | $0.7111$ | $0.6720$ | $0.6910$ | $0.8259$ | $1.8\text{ ms}$ |

---

## 5. HMM & Viterbi Sequence Reconstruction

### Emission Probability $P(z_t \mid c_{t, i})$
Blends the calibrated ML match score $P_{ML}(match \mid \mathbf{x})$ with the spatial Gaussian prior:
$$P(z_t \mid c_{t, i}) = (1 - w_{ML}) \cdot \frac{1}{\sqrt{2\pi}\sigma_z} \exp\left(-\frac{d_{\perp}^2}{2\sigma_z^2}\right) + w_{ML} \cdot P_{ML}(match \mid \mathbf{x}) \cdot \frac{1}{\sqrt{2\pi}\sigma_z}$$
Where $\sigma_z = 12.0\text{ m}$ and $w_{ML} = 0.60$.

### Transition Probability $P(c_{t, j} \mid c_{t-1, i})$
Penalizes discrepancy between road network shortest path distance and great-circle GPS displacement:
$$d_{network} = \text{DijkstraShortestPath}(\text{target}(c_{t-1, i}), \text{source}(c_{t, j}))$$
$$\Delta d = |d_{network} - d_{GPS}(z_{t-1}, z_t)|$$
$$P(c_{t, j} \mid c_{t-1, i}) = \frac{1}{\beta} \exp\left(-\frac{\Delta d}{\beta}\right)$$
Where $\beta = 25.0\text{ m}$.

### Viterbi Log-Space Dynamic Programming
$$V_{t, j} = \log P(z_t \mid c_{t, j}) + \max_{i} \left[ V_{t-1, i} + \log P(c_{t, j} \mid c_{t-1, i}) \right]$$
Backtracking produces the globally optimal sequence of candidate road segments.

### Performance Gain: ML-Only vs. HMM/Viterbi

| Metric | ML-Only Top-1 Prediction | HMM + Viterbi Reconstructed | Absolute Gain |
| :--- | :---: | :---: | :---: |
| **Recall** | $73.52\%$ | **97.52%** | **+24.00%** |
| **F1-Score** | $0.7122$ | **0.8591** | **+0.1469** |
| **Jaccard Overlap** | $55.30\%$ | **75.29%** | **+19.99%** |
| **Route Continuity** | Fragmented / Gaps | **100% Continuous Path** | Resolved via Dijkstra Bridging |
