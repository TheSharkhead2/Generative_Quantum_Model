include("../src/LinearAlgebraGenerativeModel.jl")

using .LinearAlgebraGenerativeModel 

A = [x for x in 1:32^2]
A = Matrix{ComplexF64}(transpose(reshape(A, 32, 32)))

println(A)
println(partial_trace(2, A))