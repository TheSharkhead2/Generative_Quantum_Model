"""
Unique permutation generator I grabbed from here: https://stackoverflow.com/questions/65051953/julia-generate-all-non-repeating-permutations-in-set-with-duplicates
As this seemed pretty fast, I just grabbed this instead of trying to come up with my own code. 

"""
function unique_permutations(x::T, prefix=T()) where T
    if length(x) == 1
        return [[prefix; x]]
    else
        t = T[]
        for i in eachindex(x)
            if i > firstindex(x) && x[i] == x[i-1]
                continue
            end
            append!(t, unique_permutations([x[begin:i-1];x[i+1:end]], [prefix; x[i]]))
        end
        return t
    end
end

"""
Generate all even-parity bitstrings of length N. Done simply by taking all even 
numbers up to N and grabbing all permutations of that many 1s and otherwise all 0s. 

"""
function generate_evenparity_bitstrings(N::Int)
    bitstrings = [] # empty vector to hold all even-parity bitstrings 

    for i ∈ 0:2:N # grab all even numbers between 0 and N (including 0 and N if N is even)
        permutations = unique_permutations([(Int.(ones(i)))..., (Int.(zeros(N-i)))...]) # grab all permutations of i 1s and N-i 0s 
        append!(bitstrings, permutations) # append all permutations to the bitstrings vector
    end # for 

    bitstrings 
end # function generate_evenparity_bitstrings
