include("../src/LinearAlgebraGenerativeModel.jl")

using LinearAlgebra, Statistics

using .LinearAlgebraGenerativeModel 

# ψ = Complex.([1/32 for _ in 1:32])

N = 8

epbitstrings = generate_evenparity_bitstrings(N)

quantumStates = bitstring2quantumstate.(epbitstrings)

quantumStatesProb = quantumStates .* 1/sqrt(length(quantumStates))

ψ = Complex.(sum(quantumStatesProb))

MPS = generate_mps(ψ, N)

allstates = [Complex.(Identity(2^N)[:,i]) for i ∈ 1:2^N]

oddProbs = []
evenProbs = []

for s ∈ allstates
    println(abs(dot(s, MPS))^2)
    if s ∈ quantumStates
        println("e")
        push!(evenProbs, abs(dot(s, MPS))^2)
    else
        println("o")
        push!(oddProbs, abs(dot(s, MPS))^2)
    end
    println()
end # for

println()
println(1/(2^(N-1)))

println()
println(mean(oddProbs))
println(mean(evenProbs))
println(maximum(oddProbs))
println(maximum(evenProbs))
println(median(oddProbs))
println(median(evenProbs))