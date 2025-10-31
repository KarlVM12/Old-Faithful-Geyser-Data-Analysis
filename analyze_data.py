# Karl Muller
import numpy as np
from matplotlib import pylab as plt
from sklearn.cluster import KMeans

# Load and plot data
data = np.loadtxt("faithful.dat", skiprows=26)
eruptions_data = data[:, 1]
waiting_data = data[:, 2]
X = np.column_stack((eruptions_data, waiting_data))
# print(X.shape)

plt.figure()
plt.scatter(eruptions_data, waiting_data)
plt.title("Old Faithful Geyser Eruption and Waiting Times")
plt.xlabel("Eruption Time (min)")
plt.ylabel("Waiting Time till Next Eruption (min)")
plt.grid(True)
# plt.show()

# setting up GMM with EM algo
random = np.random.default_rng(42)
n, d = X.shape
K = X.shape[1]

mu = X[random.choice(n, K, replace=False)].copy()
mu_history = [mu.copy()]

variances = np.tile(np.var(X, axis=0), (K, 1))  # assuming diag covariance
phi_mix_weights = np.array([0.5, 0.5], dtype=float)
# print(f"{mu} \n\n\n {variances} \n\n\n {phi_mix_weights}")

# Termination Criteria
threshold = 1e-6  # termination criteria: when the difference between the current EM step log likelihood and previous is less than 0.000001, means its probably not go to get any better. EM is gauranteed not to decrease with each iteration so this works on seeing if the current increase is too tiny --> possible local optimum
max_iterations = 500  # termination critiera: 500 should ideally converge by that many steps, even though on this dataset, algo seems to converge after only 7 iterations
prev_log_likelihood = -np.inf

# EM algo
for iter in range(max_iterations):
    # E step:
    # need to go through for each k at each iter
    # ( pi_k * N(x | mu_k, sigma_k) ) / ( (pi_1 * N(x | mu_1, sigma_1)) + (pi_2 * N(x | mu_2, sigma_2)) )
    gamma = np.zeros((n, K))
    for k in range(K):
        # N(x | mu_k, sigma_k) = (1 / sqrt((2pi)^d * theta(var))) exp( -0.5 * sigma_j((x_j - mu_kj)^2/(var_kj)))  # assuming diag covariance
        gaussian_likelihood = 1.0 / np.sqrt((2.0 * np.pi) ** d * np.prod(variances[k])) * np.exp(-0.5 * np.sum(((X-mu[k]) ** 2) / (variances[k]), axis=1))
        gamma[:, k] = phi_mix_weights[k] * gaussian_likelihood  # / ( (phi_mix_weights[0] * (1.0 / np.sqrt((2.0 * np.pi) ** d * np.prod(variances[0])) * np.exp(-0.5 * np.sum(((X-mu[0]) ** 2) / (variances[0]), axis=1)) ) ) + (phi_mix_weights[1] * (1.0 / np.sqrt((2.0 * np.pi) ** d * np.prod(variances[1])) * np.exp(-0.5 * np.sum(((X-mu[1]) ** 2) / (variances[1]), axis=1)) ) ) ) 
    
    gamma = gamma / gamma.sum(axis=1, keepdims=True)  # makes sure rows still sum to 1 so don't have to do full marginal calculation again for each k on gamma
    
    # M step:
    # N_k = sum_i(gamma_ik)
    # phi_k = N_k / n
    # variances = (1 / N_k) * sum_i(gamma_ik * (x_j - mu_kj)^2) w/ diag covariance
    # mu_k = (1/ N_k) sum_i(gamma_ik * x)
    N_k = gamma.sum(axis=0)
    phi_mix_weights = N_k / n
    mu = (gamma.T @ X) / N_k[:, None]
    variances = ((gamma.T @ (X ** 2)) / N_k[:, None]) - (mu ** 2)
    
    mu_history.append(mu.copy())

    # log likelihood
    # need mixture density for each point = sum_k(phi_k * N(x | mu, var))
    mix = np.zeros(n)
    epsilon = 1
    for k in range(K):
        # N(x | mu_k, sigma_k) = (1 / sqrt((2pi)^d * theta(var))) exp( -0.5 * sigma_j((x_j - mu_kj)^2/(var_kj)))  # assuming diag covariance
        gaussian_likelihood = 1.0 / np.sqrt((2.0 * np.pi) ** d * np.prod(variances[k])) * np.exp(-0.5 * np.sum(((X-mu[k]) ** 2) / (variances[k]), axis=1))
        mix += phi_mix_weights[k] * gaussian_likelihood
    
    epsilon = 1e-300  # for if ever get log(0)
    log_likelihood = np.sum(np.log(mix + epsilon))
    if np.isfinite(prev_log_likelihood) and abs(log_likelihood - prev_log_likelihood) < threshold:
        break

    prev_log_likelihood = log_likelihood
# print(f"last iter: {iter}")

# Plotting mean history throughout EM run
mu_history = np.array(mu_history)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.4, label="Data")
colors = ["tab:red", "tab:orange"]
plt.xlabel("Eruption Time (min)")
plt.ylabel("Waiting Time till Next Eruption (min)")
plt.title("Trajectory of Mean Vectors during EM run for Old Faithful")
# print(mu_history)
for k in range(mu_history.shape[1]):
    mu_x = mu_history[:, k, 0]
    mu_y = mu_history[:, k, 1]
    plt.plot(mu_x, mu_y, marker="^", color=colors[k], alpha=0.6, label=f"μ{k+1} path")
    plt.scatter(mu_x[-1], mu_y[-1], color=colors[k], s=100, alpha=0.5, edgecolor="black")

# plt.grid(True)
plt.legend()
# plt.show()

# See difference with K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X)
plt.figure(figsize=(8, 6))
plt.scatter(X[:,0], X[:,1], c=labels, cmap='autumn', s=20)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='black', marker='^', s=100, label='k-means centers')
plt.legend()
plt.xlabel("Eruption Time (min)")
plt.ylabel("Waiting Time till Next Eruption (min)")
plt.title("K-means clustering for Old Faithful")
plt.show()
# print(kmeans.cluster_centers_)
# print(mu_history)

# Running the clustering with k-means clustering instead of EM, we will get very similar results, as is shown by the graphs, but not completely identical. In this instance, it is almost like the results didn't change between the data in this dataset is very well separated so they they converge to similar solutions.
# output of k-means centers:
# [[ 4.29793023 80.28488372]
#  [ 2.09433    54.75      ]]
# vs the last EM step means:
# [[ 4.29107087 79.98562587]
#   [ 2.03791612 54.49295882]]
# They are very close, but not identical. This slight difference stems from the fact that EM probabilistically assigns points to a cluster meanwhile k-means assigns points directly to a cluster
