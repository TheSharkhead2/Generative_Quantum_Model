module LinearAlgebraGenerativeModel

using LinearAlgebra

Identity(n::Int) = Matrix{Int}(I, n, n)

export Identity

include("evenParityBitstrings.jl")

export generate_evenparity_bitstrings, bitstring2quantumstate

include("vectorspace.jl")

export outer_product, partial_trace

include("bitstringsAlgorithm.jl")

export generate_mps

end # module LinearAlgebraGenerativeModel
