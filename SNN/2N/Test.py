import numpy as np
from scipy.stats import norm

def fisher_z(r):
    return 0.5 * np.log((1 + r) / (1 - r))

def compare_correlations(r1, r2, N):
    # 1) Transform correlation coefficients to Fisher z-scores
    z1 = fisher_z(r1)
    z2 = fisher_z(r2)
    # 2) Compute standard error for the difference
    se = np.sqrt(2 / (N - 3))
    # 3) Compute Z statistic for comparing two independent correlations
    Z = (z2 - z1) / se
    # 4) Two‐tailed p-value from the standard normal distribution
    p = 2 * (1 - norm.cdf(abs(Z)))
    return Z, p

# Example usage
N  = 75992       # The size of dataset 
r1 = 0.5593      # correlation before freezing
r2 = 0.6010      # correlation after freezing

Z, p = compare_correlations(r1, r2, N)
print(f"Z = {Z:.3f}, p = {p:.3e}")