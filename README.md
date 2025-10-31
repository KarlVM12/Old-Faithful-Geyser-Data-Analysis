# Old-Faithful-Geyser-Data-Analysis
Utilizing Guassian Mixture Models and K-Means Clustering to analysis Geyser timings (eruption &amp; waiting times)

### To View Results
```bash
python -m venv venv

source venv/bin/activate

pip install -r requirement.txt

python analyze_data.py
```

### Results

<img src="./plots/plotted_data.png" alt="Old Faithful Plotted Data, Waiting Time vs Eruption Time" width="500"/>
<img src="./plots/GMM_mean_trajectory_over_time.png" alt="Trajectory of Mean Vectors during EM run for Old Faithful" width="500"/>
<img src="./plots/KMeans_clustering.png" alt="K-means clustering for Old Faithful" width="500"/>
