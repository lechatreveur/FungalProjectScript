import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm

# -------------------------------
# 1. Simulate a Toy 2D Data Set
# -------------------------------
np.random.seed(0)

# Number of points per state
n1 = 300  # State 1: Uniform
n2 = 300  # State 2: Gaussian
n3 = 0  # State 3: Mixture of Uniform and Gaussian

# State 1: Uniform in [0, 100]^2
data1 = np.random.uniform(0, 100, size=(n1, 2))

# State 2: Gaussian with mean [50, 50], sigma=5
data2 = np.random.multivariate_normal(mean=[50, 50], cov=25*np.eye(2), size=n2)

# State 3: Mixture: with probability p_true generate Gaussian, else uniform.
p_true = 0.7  # True mixing fraction for the Gaussian part in state 3.
data3 = np.empty((n3, 2))
mask3 = np.random.rand(n3) < p_true
# For Gaussian part: mean [70,70], sigma=3
data3[mask3] = np.random.multivariate_normal(mean=[50, 50], cov=25*np.eye(2), size=np.sum(mask3))
# For uniform part: uniform in [0,100]^2
data3[~mask3] = np.random.uniform(0, 100, size=(np.sum(~mask3), 2))

# Combine data and true labels
X = np.vstack((data1, data2, data3))
true_labels = np.array([0]*n1 + [1]*n2 + [2]*n3)  # 0: state1, 1: state2, 2: state3

# -------------------------------
# 2. Define the Component PDFs
# -------------------------------

def pdf_uniform(X):
    """Uniform density over [0,100]^2."""
    return np.full(X.shape[0], 1/10000)

def pdf_gaussian(X, mu, sigma):
    """Isotropic 2D Gaussian PDF."""
    diff = X - mu
    exponent = -0.5 * np.sum(diff**2, axis=1) / (sigma**2)
    return 1/(2*np.pi*sigma**2) * np.exp(exponent)

# For state 3, we define its density as a mixture:
# f_3(x) = p3 * Gaussian(x | mu3, sigma3) + (1 - p3) * Uniform(x)
def pdf_state3(X, mu3, sigma3, p3):
    f_gauss = pdf_gaussian(X, mu3, sigma3)
    f_unif = pdf_uniform(X)
    return p3 * f_gauss + (1 - p3) * f_unif

# -------------------------------
# 3. EM Algorithm for the 3-State Mixture
# -------------------------------

def e_step(X, pi, mu2, sigma2, mu3, sigma3, p3):
    """
    E-step: Compute responsibilities for each pixel for 3 states.
    
    State 1: Uniform.
    State 2: Gaussian (parameters mu2, sigma2).
    State 3: Mixture: p3 * Gaussian (mu3, sigma3) + (1-p3)*Uniform.
    """
    f1 = pdf_uniform(X)
    f2 = pdf_gaussian(X, mu2, sigma2)
    f3 = pdf_state3(X, mu3, sigma3, p3)
    
    # Total likelihood for each pixel:
    p_total = pi[0]*f1 + pi[1]*f2 + pi[2]*f3 + 1e-15
    gamma1 = pi[0]*f1 / p_total
    gamma2 = pi[1]*f2 / p_total
    gamma3 = pi[2]*f3 / p_total
    return gamma1, gamma2, gamma3

def m_step(X, gamma1, gamma2, gamma3, mu3_old, sigma3_old, p3_old):
    """
    M-step: Update mixing weights and Gaussian parameters.
    
    For state 2 and state 3, we force the Gaussian parameters to be identical.
    That is, we update a single mu_gauss and sigma_gauss based on:
      - All of gamma2 (state 2, which is Gaussian), and
      - The effective gamma from state 3 that comes from its Gaussian component.
    
    Additionally, we update p3 (the mixing fraction for the Gaussian part in state 3).
    
    Returns:
      pi_new, mu2_new, sigma2_new, mu3_new, sigma3_new, p3_new.
    """
    N = X.shape[0]
    # Update mixing weights for each state (using pixel-level responsibilities):
    pi1_new = np.mean(gamma1)
    pi2_new = np.mean(gamma2)
    pi3_new = np.mean(gamma3)
    pi_new = np.array([pi1_new, pi2_new, pi3_new])
    
    # Update Gaussian parameters for state 2 and state 3 jointly.
    # For state 2, all of gamma2 comes from the Gaussian component.
    # For state 3, f3(x) = p3_old * Gaussian(x|mu3_old, sigma3_old) + (1-p3_old)*Uniform(x)
    # so the effective weight for the Gaussian part is:
    f_gauss3 = pdf_gaussian(X, mu3_old, sigma3_old)
    f_unif = pdf_uniform(X)
    r = (p3_old * f_gauss3) / (p3_old * f_gauss3 + (1 - p3_old) * f_unif + 1e-15)
    effective_gamma_state3 = gamma3 * r  # effective responsibility for the Gaussian part of state 3
    
    # Combine state 2 and state 3 contributions:
    effective_gamma_total = gamma2 + effective_gamma_state3
    mu_gauss_new = np.sum(effective_gamma_total[:, None] * X, axis=0) / (np.sum(effective_gamma_total) + 1e-15)
    sigma_gauss_new = np.sqrt(np.sum(effective_gamma_total * np.sum((X - mu_gauss_new)**2, axis=1)) / (2*np.sum(effective_gamma_total) + 1e-15))
    
    # Constrain: state 2 and state 3 use the same Gaussian parameters:
    mu2_new = mu_gauss_new
    sigma2_new = sigma_gauss_new
    mu3_new = mu_gauss_new
    sigma3_new = sigma_gauss_new
    
    # Update p3 for state 3: the new mixing fraction is the average effective fraction
    p3_new = np.sum(gamma3 * r) / (np.sum(gamma3) + 1e-15)
    
    return pi_new, mu2_new, sigma2_new, mu3_new, sigma3_new, p3_new


# -------------------------------
# 4. Run the EM Algorithm
# -------------------------------
# Initialize parameters:
pi = np.array([0.33, 0.33, 0.34])
mu2 = np.array([40, 40])
sigma2 = 10.0
mu3 = np.array([80, 80])
sigma3 = 10.0
p3 = 0.5  # initial mixing fraction for state 3's Gaussian part

max_iter = 100
tol = 1e-4
for it in range(max_iter):
    gamma1, gamma2, gamma3 = e_step(X, pi, mu2, sigma2, mu3, sigma3, p3)
    pi_new, mu2_new, sigma2_new, mu3_new, sigma3_new, p3_new = m_step(X, gamma1, gamma2, gamma3, mu3, sigma3, p3)
    diff = (np.linalg.norm(pi_new - pi) +
            np.linalg.norm(mu2_new - mu2) +
            abs(sigma2_new - sigma2) +
            np.linalg.norm(mu3_new - mu3) +
            abs(sigma3_new - sigma3) +
            abs(p3_new - p3))
    pi, mu2, sigma2, mu3, sigma3, p3 = pi_new, mu2_new, sigma2_new, mu3_new, sigma3_new, p3_new
    if diff < tol:
        print(f"EM converged at iteration {it}")
        break

print("Estimated mixing weights (pi):", pi)
print("Estimated parameters for state 2 (Gaussian): mu2 =", mu2, ", sigma2 =", sigma2)
print("Estimated parameters for state 3 (Gaussian part): mu3 =", mu3, ", sigma3 =", sigma3)
print("Estimated mixing fraction for state 3 (Gaussian part): p3 =", p3)

# -------------------------------
# 5. Assign States and Create Masks
# -------------------------------
gamma_all = np.vstack((gamma1, gamma2, gamma3)).T  # shape (600,3)
state_assignments = np.argmax(gamma_all, axis=1)


# Assume X is the (N,2) array of 2D pixel coordinates,
# and state_assignments is an array of shape (N,) with values 0, 1, or 2,
# where 1 and 2 correspond to the Gaussian components.

# Plot all pixels in light gray:
plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], color='lightgray', s=30, label='All pixels')

# Create a mask for pixels assigned to state 2 (Gaussian) or state 3 (mixture with Gaussian)
# (Depending on your labeling, adjust the indices accordingly.)
mask = (state_assignments == 1) | (state_assignments == 2)

# Overlay red dots for these pixels:
plt.scatter(X[mask,0], X[mask,1], color='red', s=30, label='Gaussian components')

plt.xlabel("x")
plt.ylabel("y")
plt.title("Pixels assigned to Gaussian-based components")
plt.legend()
plt.show()
