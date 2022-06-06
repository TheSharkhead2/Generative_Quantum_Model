# Linear Algebra Generative Model

This project is essentially my final project for my high school linear algebra class. Contruct a model similar to the model described [here](https://iopscience.iop.org/article/10.1088/2632-2153/ab8731/pdf) and [here](https://arxiv.org/pdf/2004.05631.pdf). As I understand this model, we can use the idea of a density operator from quantum logic in order to recover information about a probability density function from samples of that PDF. This allows us to "train" our model on something like English language and construct a model that can produce English language back for us, or at least that is the idea. 

For this project, I plan on writing this as I go. So this introduction is the first thing I have written (I might revisit it if it turns out to be utterly stupid) and I will outline my process of developing an algorithm to construct this model and my learnings along the way. As a side note, in order to have this work nicely with GitHub, I have to give up LaTeX... Which for a math project is rather unfortunate. Hopefully I don't completely regret this later, but either way please work with my questionable unicode-math equations that I will probably have to use. 

## Even-Parity Bit Strings 

The example model outlined in "[At the Interface of Algebra and Statistics](https://arxiv.org/pdf/2004.05631.pdf)" by Tai-Danae Bradley, and more specifically in "[Modeling sequences with quantum states: a look under the hood](https://iopscience.iop.org/article/10.1088/2632-2153/ab8731/pdf)," looked at constructing a generative model for even-parity bit strings of length N. So, in order to have the best chance of following along, I am going to start with that as well. 

So what are the data we are working with? Well we are working with bitstrings, so essentially just strings of 1s and 0s. Something like this: 0101110. Or this: 100010. We can split these bitstrings into two categories: even if there are an even number of 1s and odd otherwise. From this, it logically follows how we construct a set of even-parity bitstrings. Essentially, all bitstrings that have an even number of 1s. As we are using this set as an example, we know the actual probability distribution of this set. In this case, just π(s) = 1/2^(N-1) if s is even and 0 otherwise. However, in a real-world example (like generating English language), we wouldn't know the *actual* probability distribution. The goal of this model is to approximate this probabilty distribution as best as possible. Essentially, we are going to start with an approximate probability distribution, ̂π, which to start we can use ̂π(s) = 1/N_T when s ∈ T (where T is the set of all bitstrings we are given, not necessarily being the complete set of even-parity bitstrings, and N_T is the number of elements in T) and 0 otherwise. 

As a bit of coding to seperate out math talk, I will quickly introduce you to a scrappy way of generating all even-parity bitstrings of length N: 
```julia 
function generate_evenparity_bitstrings(N::Int)
    bitstrings = [] # empty vector to hold all even-parity bitstrings 

    for i ∈ 0:2:N # grab all even numbers between 0 and N (including 0 and N if N is even)
        permutations = unique_permutations([(Int.(ones(i)))..., (Int.(zeros(N-i)))...]) # grab all permutations of i 1s and N-i 0s 
        append!(bitstrings, permutations) # append all permutations to the bitstrings vector
    end # for 

    bitstrings 
end # function generate_evenparity_bitstrings
```

Essentially, I loop through all even numbers between 0 and N, find all the possible permutations of that many 1s in a vector of length N, and then add that to a vector to keep track of things. At the end, I have a vector containing lists that all have an even number of 1s and represent all possible combinations of all possible counts of those 1s. 

Okay but back to the model. We want to represent these bitstrings not as bitstrings (or well my code makes bit vectors), but instead as quantum states. Now that I have thrown in the word "quantum" everyone is going to run away (though maybe they already did and I should just assume the people that stayed know what this means). In any event, some mild explaination is probably necessary. 

We can think of a quantum bit of information, or a "qubit," as a simple vector in ℂ². Now without getting into the ins and outs of quantum mechanics generally, because we don't really care for this purpose, we want to orthogonal vectors to be our on and off states. For this purpose, we can pick [1, 0] and [0, 1], which are normally refered to as "down" and "up," |d⟩ and |u⟩, or just |0⟩ and |1⟩ (I have also now just introduced bra-ket notation. Don't be scared, this: |ψ⟩ is literally just a vector. We could put a little arrow over the ψ (which I can't do with unicode), but for many reasons it is actually really nice to use bra-ket notation and is standard in quantum mechanics). 

Now, instead of representing a bitstring like so: s = 01001 ∈ T, we will represent it like so: |s⟩ = |0⟩ ⊗ |1⟩ ⊗ |0⟩ ⊗ |0⟩ ⊗ |1⟩ ∈ V^(⊗5). Here the "⊗" is the tensor product, which shows up a lot in quantum mechanics. Essentially, we went from representing each of the bits as 1 qubit, so 1 is represented with the qubit in the state |1⟩ = [0, 1], to a combined state of 5 qubits by taking their collective tensor product. There is so much to talk about to justify why this creates a combined state, but I have already derailed this way to much. I think a really good intuition for this is looking at how tensor products play with dimension. So notice how the dimension of our vector space was ⊗5. What does this mean, well essentially the dimension created by tensoring together 5 2-vectors. What is this going to be? Well for two 2-vectors tensored together, we get a vector of dimension 4 (this is because the tensor product can be seen as v ⊗ w = [w[1] ⋅ v, w[2] ⋅ v]). If we then tensored that with another 2-vector, we get a vector of dimension 8. It turns out, that the dimension of a vector that is the tensor product of n 2-vectors is going to be 2ⁿ. So, when we say the vector space has dimension ⊗5, we are saying it has dimension 2⁵, at least in this case. Why does this justify tensor products as a means to represent a combined state of qubits, well say you had 5 bits that could be 0 or 1. How many combinations of 0s and 1s are possible? Well, 2^5. Therefore, to fully represent 5 qubits which each are represented with 2 dimensions, we need 2⁵ for all 5. But now I have really gone off course, I am not trying to explain quantum mechanics here. 

With our quantum state representation of our bitstrings, we can think of the probability distribution ̂π as a unit vector in V^(⊗N) defined as: 
``` 
|ψ⟩ = 1/√N_T ⋅ ∑ |s⟩ 
```

Again, appologies for the budget unicode-math. This vector will be the input to our algorithm that will output a new vector, so called |ψₘₚₛ⟩, which is much closer to the actual distribution of the data. In particular, we aim to find the property that: |⟨s|ψₘₚₛ⟩|² ≈ π(s) = 1/2ᴺ⁻¹ for all even-parity bitstrings s. 

