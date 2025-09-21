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


   $$\displaystyle H = -h_{xx}\sum_{n=0}^{N-2}X_{n}X_{n+1} - h_{z} \sum_{n=0}^{N-1}Z_{n}$$ , \
    where $$X, Z$$ are the Pauli matrices $$\sigma^{x}, \sigma^{z}$$.
    

### 2. Yang-Lee model

**Hamiltonian:**


   $$\displaystyle H = -h_{xx}\sum_{n=0}^{N-2}X_{n}X_{n+1} - h_{z} \sum_{n=0}^{N-1}Z_{n} - +ih_{x} \sum_{n=0}^{N-1}X_{n}$$ 
   

### 3. Hubbard model with next-nearest neighbors
**Hamiltonian:**


   $$\displaystyle H = -t_{1}\sum_{j=1 \atop\sigma=\left \lbrace \uparrow,\downarrow \right\rbrace}^{L-1} \left( c_{j,\sigma}^{\dagger} c_{j+1,\sigma} + c_{j+1,\sigma}^{\dagger}c_{j,\sigma} \right) -t_{2} \sum_{j=1 \atop\sigma=\left \lbrace \uparrow,\downarrow \right\rbrace}^{L-2} \left( c_{j,\sigma}^{\dagger} c_{j+2,\sigma} + c_{j+2,\sigma}^{\dagger} c_{j,\sigma} \right) + U \sum_{j=1}^{L} n_{j,\uparrow}n_{j,\downarrow}$$ ,\
    where $$n_{j,\sigma}=c_{j,\sigma}^{\dagger} c_{j,\sigma}$$. We can always choose $$t_{1}>0$$, because we can map $$t_{1} \to -t_{1}$$ due to the gauge transformation $$c_{j,\sigma} \to e^{i\pi j}c_{j,\sigma}$$.
    

## 🚀 Getting Started

Each example is self-contained in its respective file or folder. After installing the required packages, you can run the scripts directly using Julia.

If you’re new to ITensors.jl or tensor networks in Julia, check out:

- [ITensors.jl documentation](https://itensor.github.io/ITensors.jl/stable/)
