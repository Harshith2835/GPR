---

# Fast Gaussian Process Regression with HODLR and Conjugate Gradient

This repository contains an implementation of high-performance Gaussian Process Regression (GPR) designed to bypass the traditional O(N^3) computational bottleneck. By utilizing **Hierarchical Off-Diagonal Low-Rank (HODLR)** matrix structures and iterative **Conjugate Gradient (CG)** solvers, this approach scales to significantly larger datasets than standard methods.

## Project Overview

Standard GPR requires the inversion of a dense N \times N covariance matrix, which is computationally prohibitive for large N. This project implements a hierarchical approach that compresses the matrix and solves the system iteratively.

### Key Features:

* **Hierarchical Decomposition:** Uses HODLR to represent the covariance matrix, reducing storage to O(N \log N) and matrix-vector products to O(N \log N).
* **Iterative Solver:** Employs the **Conjugate Gradient (CG)** method for linear solves, avoiding explicit matrix inversion.
* **Multiple Compression Schemes:** Includes implementations for both **SVD** (Singular Value Decomposition) and **ACA** (Adaptive Cross-Approximation).
* **Multi-Dimensional Support:** Includes 1D (index-based split), 2D/3D (KD-Tree spatial split), and N-D (GPU-accelerated) versions.

## What We Did

* **Iterative Engine:** We integrated a Conjugate Gradient solver that leverages the HODLR `matvec` operation. This allows us to solve (K + \sigma^2I)\alpha = y in O(N \log N) per iteration.
* **Spatial Trees:** For higher dimensions, we implemented KD-Trees to ensure off-diagonal blocks represent spatially distant points, maintaining the low-rank property.
* **ACA Integration:** We implemented the Partially Pivoted ACA algorithm to speed up the matrix "build" phase compared to traditional SVD.

## How to Run

1. **Dependencies:**
```bash
pip install numpy tensorflow matplotlib scipy scikit-learn
# For N-D GPU support:
pip install cupy

```


2. **Notebooks:**
* `hodlr-1d-svd.ipynb`: Basic 1D implementation using SVD.
* `hodlr-1d-aca.ipynb`: 1D implementation using the faster ACA compression.
* `HODLR_N_D.ipynb`: Advanced version for N-Dimensional data with GPU support.


3. **Execution:** Simply run the cells in the Jupyter notebooks. Each notebook contains benchmarks for "Build Time" and "RMSE" to visualize the performance gains.

---
