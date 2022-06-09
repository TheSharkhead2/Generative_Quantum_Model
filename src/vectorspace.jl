"""
Outer product. Essentially computing |ψ⟩⟨ψ|

"""
function outer_product(ψ_1, ψ_2)
    ψ_1 * transpose(ψ_2) 
end # function outer_product

"""
Partial trace calculation for tensor factorization of density operator 
for the last ⊗N-a dimensions (last N-a qubits for base 2). Different dimension
values for different basis dimensions   

"""
function partial_trace(a::Int, ρ::Matrix{ComplexF64}; base = 2)
    partialTrace = Matrix{Complex}(zeros(base^a, base^a)) # initialize partial trace matrix

    for index ∈ CartesianIndices(partialTrace) # loop through every index in partial trace (every basis combination for other space)
        for α ∈ 1:div(size(ρ)[1], size(partialTrace)[1]) # loop through the required dimension of the space we are taking a partial trace of
            partialTrace[index] += ρ[index[1]*α, index[2]*α] # add the corresponding density operator to the partial trace matrix
        end # for
    end # for

    partialTrace
end # function partial_trace