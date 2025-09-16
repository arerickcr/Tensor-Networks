# This code implements DMRG for the transverse field Ising model with Hamiltonian:
# H = -hxx * sum_n(X_{n} * X_{n+1}) - hz * sum_n(Z_n)
# where J is the coupling strength and h is the transverse field.

using ITensors, ITensorMPS
using ITensors.Strided
using LinearAlgebra
using Printf
using Random
using HDF5
using Plots

function chrd_dist(j,L)
  return 2*L/π * sin(π/L * j) 
end

#Calculate the entanglement entropy vs subinterval length
function EEcalc(psi_input)
  psi = copy(psi_input)#make a copy to not alter the original
  L = length(psi)
  eevals =  zeros(L-1)#Create a zero vector to store the entanglement entropy values. Initialized to zero
  for r=2:L-1 #from first point till last
      orthogonalize!(psi,r)#shifting orthogonality centre to position r
      #performing singular value decomposition
      U,S,V = svd(psi[r], (linkind(psi, r-1), siteind(psi,r)))
      for n=1:dim(S, 1)
          p = S[n,n]^2 #square of the diagonal elements give p_n
          eevals[r] -= p * log(p)
      end
  end
  return eevals, map(r -> chrd_dist(r, L), 1:L-1)
end

# Function that calculates the direct sum between two tensors
function directsum_MPOs(MPO1, MPO2)
  N = length(MPO1)
  all_tags = tags.(linkinds(MPO1))
  # @show all_tags
  tensors = Vector{ITensor}(undef,N)
  ind_in = undef
  for (T1, T2, i) in zip(MPO1, MPO2, 1:N)
      tags = all_tags[(i==1 ? 1 : i-1):(i==N ? N-1 : i)]
      inds1 = [filterinds(T1, tags=tt)[1] for tt in tags]
      inds2 = [filterinds(T2, tags=tt)[1] for tt in tags]
      T_out, inds_out = directsum(T1 => Tuple(inds1), T2 => Tuple(inds2), tags = tags)
      if i > 1
          replaceind!(T_out, inds_out[1], dag(ind_in))
      end
      ind_in = inds_out[end]
      tensors[i] = T_out
  end
  return MPO(tensors)
end

# Observer function to stop DMRG once a treshold is reached
mutable struct DemoObserver <: AbstractObserver
   energy_tol::Float64
   last_energy::Float64

   DemoObserver(energy_tol=0.0) = new(energy_tol,1000.0)
end

#Function that defines the Observer (checks the convergence of each sweep and stops once the tolerance is reached)
function ITensorMPS.checkdone!(o::DemoObserver;kwargs...)
  sw = kwargs[:sweep]
  energy = kwargs[:energy]
  if abs(energy-o.last_energy)/abs(energy) < o.energy_tol
    println("Stopping DMRG after sweep $sw, with energy convergence < ", o.energy_tol)
    return true
  end
  # Otherwise, update last_energy and keep going
  o.last_energy = energy
  return false
end

function ITensorMPS.measure!(o::DemoObserver; kwargs...)
  energy = kwargs[:energy]
  sweep = kwargs[:sweep]
  bond = kwargs[:bond]
  outputlevel = kwargs[:outputlevel]

  #if outputlevel > 0
  #  println("Sweep $sweep at bond $bond, the energy is $energy")
  #end
end

function Ising(N,J,hz,k,nsweeps,maxdim,psi0_init,Hz,Hxx,num_states)

  weight = 1000       # Weight for excited states, defined as H' = H + w |psi0><psi0|
                            # w has to be at least bigger than the energy gap 

  H = directsum_MPOs(hz*Hz, J*Hxx)

  # Set maximum error allowed when adapting bond dimensions
  cutoff = [1E-18]

  # If DMRG is far from the global minumum then there is no guarantee that DMRG will be able to find the true ground state. 
  # This problem is exacerbated for quantum number conserving DMRG where the search space is more constrained.
  # If this happens, a way out is to turn on the noise term feature to be a very small number.
  noise = [1E-15]

  eigsolve_krylovdim = 15 # Maximum dimension of Krylov space to locally solve problem. Try setting to a higher
                          #     value if convergence is slow or the Hamiltonian is close to a critical point.
  eigsolve_maxiter = 10    # Number of times Krylov space can be rebuild

  ishermitian = true

  psi_states = Array{Any}(undef,num_states)
  energies = Array{Any}(undef,num_states)
  Symm = Array{Any}(undef,num_states)

  obs = DemoObserver(1E-10)
  for i in 1:num_states
    if i == 1
    energy0, psi0 = dmrg(H, psi0_init; nsweeps, maxdim, cutoff, noise, observer=obs,eigsolve_krylovdim, eigsolve_maxiter, ishermitian)
    psi_states[i] = psi0
    energies[i] = energy0
    
    else
    energy1, psi1 = dmrg(H,[psi_states[j] for j in 1:(i-1)],psi0_init; nsweeps, maxdim, cutoff, noise, observer=obs, weight, eigsolve_krylovdim, eigsolve_maxiter, ishermitian)
    psi_states[i] = psi1
    energies[i] = energy1
    end
  end

  #psi_val = psi_states[1];
  #eeval, chordlen = EEcalc(psi_val);    # Entanglement entropy calculation
  
  #Check the eigenvalues of the states with respect to the Z2 symmetry of flipping all spins
  Str=OpSum();
  Str=1,"Id",1;
  Sop = Str[1];
  for i=1:N
    S=OpSum();
    S+=2,"Sx",i;
    Sop = Sop * S[1];  #Create the Z2 operator which is a product of all X_i operators 
  end
  Sx = MPO(Sop, sites);
  for i in 1:num_states
     Symm[i] = inner(psi_states[i]',Sx,psi_states[i]);
  end

  return (energies, psi_states, Symm)
end

N = 20    # Number of lattice sites
J = 1;
k = 0;      # k=0 for OBC or k=1 for PBC
num_states = 5  #Set number of desired states

# Plan to do nsweeps DMRG sweeps:
nsweeps = 10

# Set maximum MPS bond dimensions for each sweep (truncation m)
maxdim = 200

sites = siteinds("S=1/2", N; conserve_szparity=true)  # Dimensions of each site (spin 1/2 in this case) 
  #Notice that eigenvalues here are +-2Sz: +1 or -1 in this case. Here we set projection command in the conserved numbers parameter.
  #The command 'conserve_szparity' represents the Z2 symmetry of flipping all spins (chain of Z_i operators)

  # Initialize the state with the quantum number (spin) desired to initialize the Lanczos algorithm
  state = ["Dn" for n in 1:N]                         # Spin = 1 
  #state = [if n>1 "Dn" else "Up" end for n in 1:N]    # Spin = -1
  psi0_init = MPS(sites, state)

  os = OpSum()     # Initialize a sum of operators for the Hamiltonian
  # Z term in the Hamiltonian
  for j in 1:N
    os += -2, "Sz", j
  end
  Hz = MPO(os, sites) 

  os = OpSum()
  # XX term Hamiltonian 
  for j in 1:(N - 1)
    os += -4, "Sx", j, "Sx", j + 1
  end
  #Periodic boundary condition term
  os += -4*k, "Sx", N, "Sx", 1
  Hxx = MPO(os, sites)  

  partition = 20
  
    for z in 1:1
        #h = 2*z/partition;
        h=1
        energies, psi, Symm = Ising(N,J,h,k,nsweeps,maxdim,psi0_init,Hz,Hxx,num_states);
        psi0_init = psi[1]

      #Check normalization of states
      println("\nCheck normalization of states")
      for i in 1:num_states
        println("<psi",string(i-1),"|psi",string(i-1),"> = ", inner(psi[i]',psi[i]))
      end

      #Check orthogonalization of states
      println("\nCheck orthogonalization of states")
      for i in 1:num_states
        for j in (i+1):num_states
          println("<psi",string(i-1),"|psi",string(j-1),"> = ", inner(psi[i]',psi[j]))
        end
      end

      println("N = ", string(N), "\nhxx = ", string(J), "\nhz = ", string(h))
      for i in 1:num_states
        println("E",string(i-1)," = ", string(energies[i]), " with Z2 symmetry ", string(Symm[i]))
      end

    end
