# This code implements DMRG for the Hubbard model with NN and NNN interactions in its fermionic representation.

using ITensors, ITensorMPS
using LinearAlgebra

#Function that performs sums of MPOs
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

#Function that calculates the spectrum with the number of states specified (included ground state)
function excited_states(num_states,H1,H2,Hn,psi0_init,nsweeps,maxdim,L,Nf,t1,t2,U)
  weight = 1000       # Weight for excited states, defined as H' = H + w |psi0><psi0|
                            # w has to be at least bigger than the energy gap   

  H = directsum_MPOs(directsum_MPOs(t1*H1,t2*H2), U*Hn)

  #H = t1*H1 + t2*H2 + U*Hn
  # Set maximum error allowed when adapting bond dimensions
  cutoff = [0]

  # If DMRG is far from the global minumum then there is no guarantee that DMRG will be able to find the true ground state. 
  # This problem is exacerbated for quantum number conserving DMRG where the search space is more constrained.
  # If this happens, a way out is to turn on the noise term feature to be a very small number.
  noise = [1E-18]

  eigsolve_krylovdim = 50 # Maximum dimension of Krylov space to locally solve problem. Try setting to a higher
                          #     value if convergence is slow or the Hamiltonian is close to a critical point.
  eigsolve_maxiter = 100    # Number of times Krylov space can be rebuild

  ishermitian = true

  psi_states = Array{Any}(undef,num_states)
  energies = Array{Any}(undef,num_states)

  obs = DemoObserver(1E-10)
  for i in 1:num_states
    if i == 1
    energy0, psi0 = dmrg(H, psi0_init; nsweeps, maxdim, cutoff, noise, observer=obs, eigsolve_krylovdim, eigsolve_maxiter, ishermitian)
    psi_states[i] = psi0
    energies[i] = energy0
    else
    energy1, psi1 = dmrg(H,[psi_states[j] for j in 1:(i-1)],psi0_init; nsweeps, maxdim, cutoff, noise, observer=obs, weight, eigsolve_krylovdim, eigsolve_maxiter, ishermitian)
    psi_states[i] = psi1
    energies[i] = energy1
    end
  end
  return (energies,psi_states)
end

#Function that creates the initial states for given number of sites L, number of particles Nf and spin
function create_initial_state(sites, L, Nf, spin)
    """
    Create an initial state with the correct quantum numbers.
    Places particles to satisfy Npart and Sz constraints.
    """
    
    # Calculate number of up and down electrons
    # Nf = N_up + N_down
    # Sz = (N_up - N_down)/2
    # Therefore: N_up = (Nf + 2*Sz)/2, N_down = (Nf - 2*Sz)/2
    
    N_up = Int((Nf + 2*spin)/2)
    N_down = Int((Nf - 2*spin)/2)
    
    if N_up + N_down != Nf || N_up - N_down != 2*spin
        error("Invalid combination of N_particles=$Nf and Sz=$spin")
    end
    
    if N_up > L || N_down > L
        error("Too many particles for $L sites")
    end
    
    # Create state vector - simple filling from left
    state = ["Emp" for n in 1:L]
    
    # Fill up electrons
    if N_up >= N_down
      for i in 1:N_up
        if i <= N_down
            state[i] = "UpDn"  # Double occupancy
        else
            state[i] = "Up"
        end
      end
    else
      for i in 1:N_down
        if i <= N_up
            state[i] = "UpDn"  # Double occupancy
        else
            state[i] = "Dn"
        end
      end
    end
    
    return productMPS(sites, state)
end



# Main function
let

#set parameters
L = 6       # number of sites
Nf = 4      # number of particles
spin = 0    # Total Sz of the system
k = 0       # k=0 for OBC or k=1 for PBC 
t1 = 1      # nn hopping coupling
t2 = -0.5      # nnn hopping coupling
U = 10       # repulsion coupling
num_states = 4 # number of states to calculate (including ground state)

# Plan to do nsweeps DMRG sweeps:
nsweeps = 30

# Set maximum MPS bond dimensions for each sweep (truncation m)
maxdim = 500

sites = siteinds("Electron", L, conserve_sz=true, conserve_nf=true, conserve_nfparity=true)

# Initialize the state with the quantum number (spin) desired to initialize the Lanczos algorithm
psi0_init = create_initial_state(sites, L, Nf, spin)

  os1 = OpSum()     # Initialize a sum of operators for the Hamiltonian
  # NN hopping term in the Hamiltonian
  for j in 1:L-1
    os1 += -1.0, "Cdagup", j + 1, "Cup", j 
    os1 += -1.0, "Cdagup", j, "Cup", j + 1
    os1 += -1.0, "Cdagdn", j + 1, "Cdn", j 
    os1 += -1.0, "Cdagdn", j, "Cdn", j + 1
  end
  #Periodic boundary condition term
  os1 += -k, "Cdagup", 1, "Cup", L
  os1 += -k, "Cdagup", L, "Cup", 1
  os1 += -k, "Cdagdn", 1, "Cdn", L
  os1 += -k, "Cdagdn", L, "Cdn", 1
  H1 = MPO(os1, sites) 

  os2 = OpSum()     # Initialize a sum of operators for the Hamiltonian
  # NNN hopping term in the Hamiltonian
  for j in 1:L-2
    os2 += -1.0, "Cdagup", j + 2, "Cup", j
    os2 += -1.0, "Cdagup", j, "Cup", j + 2
    os2 += -1.0, "Cdagdn", j + 2, "Cdn", j
    os2 += -1.0, "Cdagdn", j, "Cdn", j + 2
  end
  #Periodic boundary condition term
  os2 += -k, "Cdagup", 2, "Cup", L
  os2 += -k, "Cdagup", L, "Cup", 2
  os2 += -k, "Cdagdn", 2, "Cdn", L
  os2 += -k, "Cdagdn", L, "Cdn", 2
  os2 += -k, "Cdagup", 1, "Cup", L - 1
  os2 += -k, "Cdagup", L - 1, "Cup", 1
  os2 += -k, "Cdagdn", 1, "Cdn", L - 1
  os2 += -k, "Cdagdn", L - 1, "Cdn", 1
  H2 = MPO(os2, sites)  

  os3 = OpSum()     # Initialize a sum of operators for the Hamiltonian
  # Repulsion term hopping term in the Hamiltonian
  for j in 1:L
    os3 += 1.0, "Nup", j, "Ndn", j
  end
  Hn = MPO(os3, sites)  

  #Calculate the spectrum
  energies,psi_states = excited_states(num_states,H1,H2,Hn,psi0_init,nsweeps,maxdim,L,Nf,t1,t2,U);

  #Check normalization of states
  println("\nCheck normalization of states")
  for i in 1:num_states
    println("<psi",string(i-1),"|psi",string(i-1),"> = ", inner(psi_states[i]',psi_states[i]))
  end

  #Check orthogonalization of states
  println("\nCheck orthogonalization of states")
  for i in 1:num_states
    for j in (i+1):num_states
      println("<psi",string(i-1),"|psi",string(j-1),"> = ", inner(psi_states[i]',psi_states[j]))
    end
  end
  
  osp = OpSum()   
  # Create operator total S+
  for j in 1:L
    osp += 1.0, "S+", j
  end
  
  osm = OpSum()     
  # Create operator total S-
  for j in 1:L
    osm += 1.0, "S-", j
  end
  
  osZ = OpSum()     
  # Create operator total Sz
  for j in 1:L
    osZ += 1.0, "Sz", j
  end

  opS2 = OpSum();
  # Create operator total S^2
  for j in 1:L
    opS2 += osp[j]*osm
    opS2 += osZ[j]*osZ
    opS2 += -1.0, "Sz", j
  end
  S2 = MPO(opS2,sites)
  Sz = MPO(osZ, sites) 
  
  # Print energies with corresponding Sz and S^2
  println("\nEnergies with corresponding Sz and S^2")
  for i in 1:num_states
    println("E",string(i-1)," = ", string(energies[i]), " with Sz: ", string(inner(psi_states[i]',Sz,psi_states[i])), " and total spin S(S+1): ", string(inner(psi_states[i]',S2,psi_states[i])))
  end

end
  
