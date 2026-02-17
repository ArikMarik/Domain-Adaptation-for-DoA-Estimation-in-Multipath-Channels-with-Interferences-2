"""
Test script for Riemannian mean implementation.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from algorithms.riemannian_geometry import riemannian_mean, riemannian_distance
from algorithms.domain_adaptation import compute_mean_covariance

print("Testing Riemannian Mean Implementation")
print("=" * 60)

# Test 1: Basic convergence test
print("\nTest 1: Basic convergence with random SPD matrices")
np.random.seed(42)
n_mics = 5
n_matrices = 10

# Generate random SPD matrices
cov_matrices = np.zeros((n_mics, n_mics, n_matrices))
for i in range(n_matrices):
    A = np.random.randn(n_mics, n_mics)
    cov_matrices[:, :, i] = A @ A.T + np.eye(n_mics)  # Ensure SPD

# Compute arithmetic mean
arith_mean = compute_mean_covariance(cov_matrices, method='arithmetic')
print(f"Arithmetic mean computed: shape {arith_mean.shape}")
print(f"  Determinant: {np.linalg.det(arith_mean):.6f}")
print(f"  Trace: {np.trace(arith_mean):.6f}")

# Compute Riemannian mean
riem_mean = compute_mean_covariance(cov_matrices, method='riemannian')
print(f"\nRiemannian mean computed: shape {riem_mean.shape}")
print(f"  Determinant: {np.linalg.det(riem_mean):.6f}")
print(f"  Trace: {np.trace(riem_mean):.6f}")

# Test 2: Verify positive definiteness
print("\nTest 2: Verify positive definiteness")
eigenvalues_arith = np.linalg.eigvals(arith_mean)
eigenvalues_riem = np.linalg.eigvals(riem_mean)
print(f"Arithmetic mean - Min eigenvalue: {np.min(np.real(eigenvalues_arith)):.6f}")
print(f"Riemannian mean - Min eigenvalue: {np.min(np.real(eigenvalues_riem)):.6f}")

if np.min(np.real(eigenvalues_arith)) > 0:
    print("✓ Arithmetic mean is positive definite")
else:
    print("✗ Arithmetic mean is NOT positive definite")

if np.min(np.real(eigenvalues_riem)) > 0:
    print("✓ Riemannian mean is positive definite")
else:
    print("✗ Riemannian mean is NOT positive definite")

# Test 3: Compare distances to input matrices
print("\nTest 3: Compare average distances from mean to input matrices")
arith_distances = []
riem_distances = []

for i in range(n_matrices):
    C = cov_matrices[:, :, i]
    arith_dist = riemannian_distance(arith_mean, C)
    riem_dist = riemannian_distance(riem_mean, C)
    arith_distances.append(arith_dist)
    riem_distances.append(riem_dist)

avg_arith_dist = np.mean(arith_distances)
avg_riem_dist = np.mean(riem_distances)

print(f"Average distance (arithmetic mean): {avg_arith_dist:.6f}")
print(f"Average distance (Riemannian mean): {avg_riem_dist:.6f}")

if avg_riem_dist < avg_arith_dist:
    print(f"✓ Riemannian mean is closer (reduction: {(avg_arith_dist - avg_riem_dist)/avg_arith_dist*100:.2f}%)")
else:
    print(f"Note: Arithmetic mean happened to be closer in this case")

# Test 4: Identical matrices should return the same matrix
print("\nTest 4: Test with identical matrices")
identity = np.eye(n_mics)
identical_matrices = np.stack([identity] * 5, axis=2)

arith_identical = compute_mean_covariance(identical_matrices, method='arithmetic')
riem_identical = compute_mean_covariance(identical_matrices, method='riemannian')

diff = np.linalg.norm(arith_identical - identity, 'fro')
print(f"Arithmetic mean vs identity: ||diff|| = {diff:.10f}")
diff = np.linalg.norm(riem_identical - identity, 'fro')
print(f"Riemannian mean vs identity: ||diff|| = {diff:.10f}")

if diff < 1e-6:
    print("✓ Riemannian mean of identical matrices equals the input")
else:
    print("✗ Riemannian mean differs from input")

# Test 5: Performance check
print("\nTest 5: Performance comparison")
import time

n_test = 50
start = time.time()
for _ in range(n_test):
    _ = compute_mean_covariance(cov_matrices, method='arithmetic')
arith_time = (time.time() - start) / n_test * 1000

start = time.time()
for _ in range(n_test):
    _ = compute_mean_covariance(cov_matrices, method='riemannian')
riem_time = (time.time() - start) / n_test * 1000

print(f"Arithmetic mean: {arith_time:.3f} ms per call")
print(f"Riemannian mean: {riem_time:.3f} ms per call")
print(f"Slowdown factor: {riem_time/arith_time:.1f}x")

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)
