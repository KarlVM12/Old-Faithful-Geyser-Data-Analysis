# Old Faithful Geyser Clustering

This project looks at the Old Faithful eruption dataset and frames it as a clustering problem tackled with a Gaussian Mixture Model trained via Expectation Maximization (EM), alongside a K-Means
baseline. The goal is to show, quantitatively and visually, how probabilistic soft assignments from EM contrast with the hard assignment regions produced by K-Means.

## Overview
- **Data:** eruption duration (minutes) and waiting time to the next eruption (minutes) collected by the USGS.
- **Models:** an implemented from scratch coded diagonal covariance GMM using EM (`analyze_data.py`) and a scikit `KMeans` baseline fit 
- **Outputs:** scatter plot of the raw data, EM mean trajectory plot showing convergence, and final K-Means clustering.

## Implementation
- **Custom EM Loop:** Initialized component means from random data points and iterated the closed form EM updates for diagonal covariances until the log likelihood improvement fell below a termination threshold of `1e-6`. The run typically converges in ≤10 iterations.
- **Soft Responsibilities:** Each E step computes responsibilities `γ_ik` that smoothly reweight the sufficient statistics, letting overlapping points contribute proportionally to both latent eruption regimes.
- **Parameter Updates:** The M step refreshes the mixing weights, means, and variances from the responsibility weighted sums, preserving the guarantee that the likelihood will not decrease.
- **Baseline K-Means:** A two class KMeans fit provided a hard assignment comparison useful for highlighting the effect of probabilistic clustering on overlapping regions.

## Results
- The EM run stabilizes at mean vectors of roughly `[4.29, 79.99]` and `[2.04, 54.49]` minutes, matching the assumption of long eruption, long wait vs short eruption, short wait.
- Final K-Means centroids `[4.30, 80.28]` and `[2.09, 54.75]` are nearly identical, underscoring that this dataset is well separated but still benefits from EM’s ability to quantify uncertainty around the decision boundary.
- The mean trajectory plot visualizes how each EM update walks the centroids directly toward their dense regions, while the K-Means figure exposes the sharp partitioning induced by hard assignments.

<img src="./plots/plotted_data.png" alt="Old Faithful Plotted Data, Waiting Time vs Eruption Time" width="500"/>
<img src="./plots/GMM_mean_trajectory_over_time.png" alt="Trajectory of Mean Vectors during EM run for Old Faithful" width="500"/>
<img src="./plots/KMeans_clustering.png" alt="K-means clustering for Old Faithful" width="500"/>

## Reproducing the Results
```bash
python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

python analyze_data.py
```
