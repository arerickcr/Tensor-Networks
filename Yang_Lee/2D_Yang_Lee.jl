# This code reproduces the spectrum of 2D Yang Lee model, which is represented as a 1D spin chain with Hamiltonian
# H = -hxx * sum_n(X_{n} * X_{n+1}) - hz * sum_n(Z_n) + ihx * sum_n(X_n)
# where J is the coupling strength and hz is the transverse magnetic field and hx is the longitudinal magnetic field.

using ITensors, ITensorMPS
using LinearAlgebra
using Printf
using Random
using HDF5
using Plots

function YL(N,J,hx,hz,k,nsweeps,maxdim)

  sites = siteinds("S=1/2", N; conserve_szparity=false)  # Dimensions of each site (spin 1/2 in this case) 
  #Notice that eigenvalues here are +-2Sz: +1 or -1 in this case. Here we set projection command in the conserved numbers parameter.
  #The command 'conserve_szparity' represents the Z2 symmetry of flipping all spins (chain of Z_i operators)

  weight = 100       # Weight for excited states, defined as H' = H + w |psi0><psi0|
                            # w has to be at least bigger than the energy gap

  os = OpSum()     # Initialize a sum of operators for the Hamiltonian

  # Z term in the Hamiltonian
  for j in 1:N
    os += -2*hz, "Sz", j
  end

  # X term in the Hamiltonian
  for j in 1:N
    os += 2im*hx, "Sx", j
  end

  # XX term Hamiltonian 
  for j in 1:(N - 1)
    os += -4*J, "Sx", j, "Sx", j + 1
  end
  #Periodic boundary condition term
  os += -4*k*J, "Sx", N, "Sx", 1

  # Create the MPO for the Hamiltonian
  H = MPO(os, sites)    

  # Initialize the state with the quantum number (spin) desired to initialize the Lanczos algorithm
  state = ["Dn" for n in 1:N]  

  # Create an MPS for the previous state
  psi0_init = MPS(sites, state)

  # Set maximum error allowed when adapting bond dimensions
  cutoff = [1E-18]

  # If DMRG is far from the global minumum then there is no guarantee that DMRG will be able to find the true ground state. 
  # This problem is exacerbated for quantum number conserving DMRG where the search space is more constrained.
  # If this happens, a way out is to turn on the noise term feature to be a very small number.
  noise = [1E-15]

  eigsolve_krylovdim = 10 # Maximum dimension of Krylov space to locally solve problem. Try setting to a higher
                          #     value if convergence is slow or the Hamiltonian is close to a critical point.
  eigsolve_maxiter = 100    # Number of times Krylov space can be rebuild

  ishermitian = false      # Notice that we have a non-Hermitian operator now

  # Run the DMRG algorithm, returning energy and optimized MPS of ground state
  energy0, psi0 = dmrg(H, psi0_init; nsweeps, maxdim, cutoff, noise, eigsolve_krylovdim, eigsolve_maxiter, ishermitian)

  # Initialize the first excited state for Lanczos
  state1 = [if n>1 "Dn" else "Up" end for n in 1:N]
  psi1_init = MPS(sites,state1)

  # Run DMRG for energy and optimized MPS for first excited state
  # Notice that we have left and right eigenstates and we have to differentiate them. In this case (because of PT symmetry) <ψ_R| = (|ψ_L>)^T or |ψ_R> = |ψ_L>^*
  energy1,psi1 = dmrg(H,[dag(psi0)],psi1_init; nsweeps, maxdim, cutoff, noise, weight, eigsolve_krylovdim, eigsolve_maxiter, ishermitian)

  # Initialize the second excited state for Lanczos
  state2 = [if n>2 "Dn" else "Up" end for n in 1:N]
  psi2_init = MPS(sites,state2)

  # Run DMRG for energy and optimized MPS for second excited state
  energy2,psi2 = dmrg(H,[dag(psi0),dag(psi1)],psi2_init; nsweeps, which_decomp, maxdim, cutoff, noise, weight, eigsolve_krylovdim, eigsolve_maxiter, ishermitian)

  # Initialize the third excited state for Lanczos
  state3 = [if n>3 "Dn" else "Up" end for n in 1:N]
  psi3_init = MPS(sites,state3)

  # Run DMRG for energy and optimized MPS for second excited state
  energy3,psi3 = dmrg(H,[dag(psi0),dag(psi1),dag(psi2)],psi3_init; nsweeps, which_decomp, maxdim, cutoff, noise, weight, eigsolve_krylovdim, eigsolve_maxiter, ishermitian)

  # Initialize the third excited state for Lanczos
  state4 = [if n>4 "Dn" else "Up" end for n in 1:N]
  psi4_init = MPS(sites,state4)

  # Run DMRG for energy and optimized MPS for second excited state
  energy4,psi4 = dmrg(H,[dag(psi0),dag(psi1),dag(psi2),dag(psi3)],psi4_init; nsweeps, which_decomp, maxdim, cutoff, noise, weight, eigsolve_krylovdim, eigsolve_maxiter, ishermitian)

  # Initialize the third excited state for Lanczos
  state5 = [if n>5 "Dn" else "Up" end for n in 1:N]
  psi5_init = MPS(sites,state5)

  # Run DMRG for energy and optimized MPS for second excited state
  energy5,psi5 = dmrg(H,[dag(psi0),dag(psi1),dag(psi2),dag(psi3),dag(psi4)],psi5_init; nsweeps, which_decomp, maxdim, cutoff, noise, weight, eigsolve_krylovdim, eigsolve_maxiter, ishermitian)

  # Initialize the third excited state for Lanczos
  state6 = [if n>6 "Dn" else "Up" end for n in 1:N]
  psi6_init = MPS(sites,state6)

  # Run DMRG for energy and optimized MPS for second excited state
  energy6,psi6 = dmrg(H,[dag(psi0),dag(psi1),dag(psi2),dag(psi3),dag(psi4),dag(psi5)],psi6_init; nsweeps, which_decomp, maxdim, cutoff, noise, weight, eigsolve_krylovdim, eigsolve_maxiter, ishermitian)

  return (energy0,energy1,energy2,energy3,energy4,energy5,energy6)
end

let
N = 10    # Number of lattice sites
J = 1
hz = 2
k = 0      # k=0 for OBC or k=1 for PBC

# Plan to do nsweeps DMRG sweeps:
nsweeps = 10

# Set maximum MPS bond dimensions for each sweep (truncation m)
maxdim = 500

partition = 10
x = zeros(partition + 1)
y = zeros(7,partition + 1)
z = zeros(7,partition + 1)

for m in 0:partition
    hx = 0.25 + 0.5*m/partition
    E0,E1,E2,E3,E4,E5,E6 = YL(N,J,hx,hz,k,nsweeps,maxdim)
    println("hx = ", string(hx),"\nE0 = ", string(E0) ,"\nE1 = ", string(E1),"\nE2 = ", string(E2), "\nE3 = ", string(E3),
    "\nE4 = ", string(E4),"\nE5 = ", string(E5),"\nE6 = ", string(E6),"\n")
    x[m+1] = hx
    y[1,m+1] = real(E0)
    y[2,m+1] = real(E1)
    y[3,m+1] = real(E2)
    y[4,m+1] = real(E3)
    y[5,m+1] = real(E4)
    y[6,m+1] = real(E5)
    y[7,m+1] = real(E6)
    z[1,m+1] = 2*real(E0 - E0)/real(E2 - E0)
    z[2,m+1] = 2*real(E1 - E0)/real(E2 - E0)
    z[3,m+1] = 2*real(E2 - E0)/real(E2 - E0)
    z[4,m+1] = 2*real(E3 - E0)/real(E2 - E0)
    z[5,m+1] = 2*real(E4 - E0)/real(E2 - E0)
    z[6,m+1] = 2*real(E5 - E0)/real(E2 - E0)
    z[7,m+1] = 2*real(E6 - E0)/real(E2 - E0)
end

plot(x,[y[1,:] y[2,:] y[3,:] y[4,:] y[5,:] y[6,:] y[7,:]], label=["\$E_{0}\$" "\$E_1\$" "\$E_2\$" "\$E_3\$" "\$E_4\$" "\$E_5\$" "\$E_6\$"], lw=2, xlabel="\$-ihx\$", ylabel = "\$Re(E_n)\$")
#plot(x,[z[1,:] z[2,:] z[3,:] z[4,:] z[5,:] z[6,:] z[7,:]], label=["\$Δ_{0}\$" "\$Δ_1\$" "\$Δ_2\$" "\$Δ_3\$" "\$Δ_4\$" "\$Δ_5\$" "\$Δ_6\$"], lw=2, xlabel="\$-ihx\$", ylabel = "\$Re(E_n)\$")
end
