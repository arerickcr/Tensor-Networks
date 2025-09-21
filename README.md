# Tensor Networks examples

Some projects that use Tensor Networks in Julia.

To start, we need to install the following packages: 

ITensors, ITensorMPS, LinearAlgebra, Arpack, SparseArrays, Plots, Random, HDF5

We do this with the command: using Pkg

Examples:
> Pkg.add("ITensors")
> 
> Pkg.add("ITensorMPS")
> 
> Pkg.add("...")

The examples shown here are the following:
1. **Transverse Ising model** \
   $$\displaystyle H = -h_{xx}\sum_{n=0}^{N-2}X_{n}X_{n+1} - h_{z} \sum_{n=0}^{N-1}Z_{n}$$ , \
    where $$X, Z$$ are the Pauli matrices $$\sigma^{x}, \sigma^{z}$$.

2. **Yang-Lee model** \
   $$\displaystyle H = -h_{xx}\sum_{n=0}^{N-2}X_{n}X_{n+1} - h_{z} \sum_{n=0}^{N-1}Z_{n} - +ih_{x} \sum_{n=0}^{N-1}X_{n}$$

3. **Hubbard model with next-nearest neighbors** \
   $$\displaystyle H = -t_{1}\sum_{j=1 \atop\sigma=\left \lbrace \uparrow,\downarrow \right\rbrace}^{L-1} \left( c_{j,\sigma}^{\dagger} c_{j+1,\sigma} + c_{j+1,\sigma}^{\dagger}c_{j,\sigma} \right) -t_{2} \sum_{j=1 \atop\sigma=\left \lbrace \uparrow,\downarrow \right\rbrace}^{L-2} \left( c_{j,\sigma}^{\dagger} c_{j+2,\sigma} + c_{j+2,\sigma}^{\dagger} c_{j,\sigma} \right) + U \sum_{j=1}^{L} n_{j,\uparrow}n_{j,\downarrow}$$ ,\
    where $$n_{j,\sigma}=c_{j,\sigma}^{\dagger} c_{j,\sigma}$$. We can always choose $$t_{1}>0$$, because we can map $$t_{1} \to -t_{1}$$ due to the gauge transformation $$c_{j,\sigma} \to e^{i\pi j}c_{j,\sigma}$$.
