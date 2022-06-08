"""
Generate MPS vector for given input data of bitstrings of length N 

"""
function generate_mps(ψ::Vector{ComplexF64}, N::Int)
    # step 1 
    M_1 = reshape(ψ, :, 2) # reshape into map V → V^(⊗N-1)

    M_1 = M_1 * Identity(2) # apply the identity to the map 

    ψ_2 = reshape(M_1, :, 1) # reshape map back into a vector

    # further steps 
    Umatrices = [] # empty vector to store all Us we create as they will be important 

    ψ_k = ψ_2 # initialize ψ_k to ψ_2 
    for _ ∈ 1:N-2 # loop N-2 times 
        partialTrace = partial_trace(2, outer_product(ψ_k, ψ_k)) # calculate partial trace of the density matrix for ψ_k for ⊗N-2
        
        U = eigvecs(partialTrace)[:, 3:4] # find eigenvectors of partial trace and remove the two with the smallest eigenvalues
    
        M_k = reshape(ψ_k, :, 4) # reshape ψ_k into a map V^(⊗2) → V^(⊗N-2)

        MU = M_k * U # apply the U to the map 

        ψ_k = reshape(MU, :, 1) # reshape the map back into a vector

        push!(Umatrices, U) # push the U into the Umatrices vector
    end # for

    # last step 
    U_N = reshape(ψ_k, :, 2) # reshape ψ_k into a map V^(⊗1) → V^(⊗1)
    
    # generate MPS 
    MPS = Identity(2^(N-1)) # initialize MPS
    for (k, U) ∈ enumerate(Umatrices) # loop through all Us 
        MPS = MPS * kron(U, Identity(2^(N-2-k))) 
    end # for

    MPS = MPS * U_N # apply the last U to the MPS

    reshape(transpose(MPS), :, 1) # reshape the MPS into a vector

end # function generate_mps
