include("../src/LinearAlgebraGenerativeModel.jl")

using .LinearAlgebraGenerativeModel 

println(generate_evenparity_bitstrings(4))

quantumState = bitstring2quantumstate([0,1,1,1])

println(quantumState)
println(digits(findall(x->x==1, quantumState)[1]-1, base=2; pad=4) |> reverse)