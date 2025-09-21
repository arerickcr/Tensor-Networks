# Tensor Networks Examples in Julia

This repository contains a collection of simple and instructive examples using **tensor networks** implemented in **Julia**, primarily with the [`ITensors.jl`](https://github.com/ITensor/ITensors.jl) library and related tools.

---

## 📦 Required Packages

Before running the examples, install the following Julia packages:

- `ITensors`
- `ITensorMPS`
- `LinearAlgebra`
- `Arpack`
- `SparseArrays`
- `Plots`
- `Random`
- `HDF5`

Install them using Julia's package manager:

```julia
using Pkg

Pkg.add("ITensors")
Pkg.add("ITensorMPS")
Pkg.add("LinearAlgebra")
Pkg.add("Arpack")
Pkg.add("SparseArrays")
Pkg.add("Plots")
Pkg.add("Random")
Pkg.add("HDF5")
``` 
Make sure you're using a recent version of Julia and ITensors.jl.

## 🧪 Examples Included

This repository contains code for simulating the following models:

### 1. Transverse Field Ising Model

**Hamiltonian:**


   $$\begin{align}
   H = -h_{xx}\sum_{n=0}^{N-2}X_{n}X_{n+1} - h_{z} \sum_{n=0}^{N-1}Z_{n}\ ,
   \end{align}$$
   
   where $$X, Z$$ are the Pauli matrices $$\sigma^{x}, \sigma^{z}$$.
    

### 2. Yang-Lee model

**Hamiltonian:**


   $$\begin{align}
   H = -h_{xx}\sum_{n=0}^{N-2}X_{n}X_{n+1} - h_{z} \sum_{n=0}^{N-1}Z_{n} - ih_{x} \sum_{n=0}^{N-1}X_{n}\ .
   \end{align}$$ 
   

### 3. Hubbard model with next-nearest neighbors
**a) Hamiltonian in fermionic formalism:**


   $$\begin{align}
   H = -t_{1}\sum_{j=1 \atop\sigma=\left \lbrace \uparrow,\downarrow \right\rbrace}^{L-1} \left( c_{j,\sigma}^{\dagger} c_{j+1,\sigma} + c_{j+1,\sigma}^{\dagger}c_{j,\sigma} \right) -t_{2} \sum_{j=1 \atop\sigma=\left \lbrace \uparrow,\downarrow \right\rbrace}^{L-2} \left( c_{j,\sigma}^{\dagger} c_{j+2,\sigma} + c_{j+2,\sigma}^{\dagger} c_{j,\sigma} \right) + U \sum_{j=1}^{L} n_{j,\uparrow}n_{j,\downarrow}\ ,
   \end{align}$$
   
   where $$n_{j,\sigma}=c_{j,\sigma}^{\dagger} c_{j,\sigma}$$. We can always choose $$t_{1}>0$$, because we can map $$t_{1} \to -t_{1}$$ due to the gauge transformation $$c_{j,\sigma} \to e^{i\pi j}c_{j,\sigma}$$.

**b) Hamiltonian in qubit formalism**


$$\begin{align}
    H & =t_{1} \sum_{k=1}^{2L-2}\left( \sigma^{+}_{k} \sigma^{z}_{k+1} \sigma^{-}_{k+2} + \sigma^{-}_{k} \sigma^{z}_{k+1} \sigma^{+}_{k+2} \right) + t_{2} \sum_{k=1}^{2L-4}\left( \sigma^{+}_{k} \sigma^{z}_{k+1} \sigma^{z}_{k+2} \sigma^{z}_{k+3} \sigma^{-}_{k+4} + \sigma^{-}_{k} \sigma^{z}_{k+1} \sigma^{z}_{k+2}\sigma^{z}_{k+3}\sigma^{+}_{k+4}\right) + \notag\\
    & \quad + \frac{U}{4} \sum_{k=1}^{L} \left( \sigma^{z}_{2k-1} +1 \right)\left( \sigma^{z}_{2k} +1 \right) .
\end{align}$$

## 🚀 Getting Started

Each example is self-contained in its respective file or folder. After installing the required packages, you can run the scripts directly using Julia.

If you’re new to tensor networks in Julia, ITensors.jl, or ITensorMPS.jl, check out:

- [ITensor webpage] (https://itensor.org/)
- [ITensors.jl documentation](https://itensor.github.io/ITensors.jl/stable/)
- [ITensorMPS.jl documentation](https://docs.itensor.org/ITensorMPS/stable/)
