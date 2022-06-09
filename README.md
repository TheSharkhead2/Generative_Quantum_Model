# Generative Model Using Quantum Logic

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

Again, apologizes for the budget unicode-math. This vector will be the input to our algorithm that will output a new vector, so called |ψₘₚₛ⟩, which is much closer to the actual distribution of the data. In particular, we aim to find the property that: |⟨s|ψₘₚₛ⟩|² ≈ π(s) = 1/2ᴺ⁻¹ for all even-parity bitstrings s. It is best to think of the |ψ⟩ vector as the training data for our model as it is the empirical probabilities observed in our data set. 

Before we get that much further into the algorithm, lets first just build some code to construct this quantum state from a bitstring. This is really not the most complicated code: 
```julia
function bitstring2quantumstate(bitstring::Vector{Int})
    qubits = [b==0 ? [1,0] : [0, 1] for b ∈ bitstring] # convert bits to qubits where 0 → |d⟩ and 1 → |u⟩

    ψ = popat!(qubits, 1) # grab the first qubit and remove it from the vector 
    for qubit ∈ qubits 
        ψ = kron(ψ, qubit) # get tensor product between the tensored last qubits and the next qubit 
    end # for

    ψ 
end # function bitstring2quantumstate
```

We, just as you would think, convert each of the bits to qubits, and then just tensor them all up in order. 

Now that we have all that random bitstring stuff out of the way, we start on constructing the actual algorithm. Our algorithm is going to be N steps, where they are inductive steps meaning all steps past step 3 are nearly identical to step 2. 

The first step to the algorithm is basically just here so that things look nicer. We basically just define the 2x2 identity operator on ℂ² which is exactly what you know it to be. Ones on the diagonal (from top left to bottom right) and zeros otherwise. 

That was a pretty lame step, but this gets more interesting at step 2. Note that this is where things really got confusing for me so hopefully I can do everything right here. 

We start the second step by reshaping our input vector, |ψ⟩, in a way such that it becomes an map from V → V^(⊗N-1), so that we can apply the identity operator (or the operator from our last step), giving us the second vector, |ψ₂⟩. Functionally this gives us the same thing as |ψ⟩, but this sets up the induction for our algorithm which is important. 

While I am still not entirely sure on this step, I believe we can do this conversion through the following code: 

```julia 
M_1 = reshape(ψ, :, 2)
```

What this simple code does is literally reshape the nx1 vector to a n/2 x 2 vector. This means we create a map from V → V^(⊗N-1), which we want. As an example, we would map the following vector like so: 
```
1
2
3                   1  6
4                   2  7
5         →         3  8
6                   4  9
7                   5  10
8
9
10

```

Forgive the poor formatting again. As a more specific description of what exactly this line is doing. ```reshape(A, n, m)``` will reshape the matrix/vector A into one of dimensions n x m. 

Once we have reshaped the vector, we can apply the identity operator from before: 

```julia 
M_1 = M_1 * Identity(2)
``` 

After we apply the matrix, we can reshape the map back into a vector: 

```julia
ψ_2 = reshape(M_1, :, 1)
```

At this point, we want to find the density operator for |ψ₂⟩ which we can think of as a quantum version of a probability distribution. However important for our situation, we can also define a density operator for a classical probability distribution and the other way around. Our vector |ψ₂⟩ can be thought of as a vector where each entry is the square root of the probability of the qubit string corresponding to that entry. It turns out, that to get the density operator for this state, we can simply compute an outer product: |ψ₂⟩⟨ψ₂|. We can implement an outer product like so: 
```julia 
function outer_product(ψ_1, ψ_2)
    ψ_1 * transpose(ψ_2) 
end # function outer_product
```

Now while this density operator is cool and all, we really want to find the reduced density operator which comes about from a partial trace. It turns out, that taking a partial trace of a density operator gives us another density operator which can be thought of as finding the marginal probability! So we can think of the density operator as encoding the probability of every bitstring occurring and when we take the partial trace to get the reduced density operator, that is akin to finding the probabilities for sub-bitstrings. We define a partial trace like so: ```∑ ρᵢₐ,ₐⱼ |xᵢ⟩⟨xⱼ|```. Here we can think of decomposing the full space into two subspaces tensored together where one has the |xᵢ⟩ basis and the other a |yₐ⟩ basis. Here we are taking the partial trace with respect to the |yₐ⟩ basis, so finding the marginal probabily (reduced density operator) for the |xᵢ⟩ basis. This is done by taking each of the states in the |xᵢ⟩ basis, so summing over all i and j, and then summing over a diagonal in the other basis (hence why we are summing over the coordinates in the full density opeartor of (i⋅a, j⋅a)). 

In our case, we know |xᵢ⟩ will just be the standard basis. This is convenient as we can essentially simplify this to constructing an d x d matrix, where d = 2ᵃ where the dimension of the space we are finding the partial trace of is ⊗N-a (here a is the number of qubits in the subspace we are finding a reduced density for, sorry confusing to which it up), and each entry is just the above sum for i, j representing the location in this matrix. 

In order to implement the partial trace in code, we can of course start by creating an empty matrix of the right size: 

```julia 
function partial_trace(a::Int, ρ::Matrix{ComplexF64}; base = 2)
    partialTrace = Matrix{Complex}(zeros(base^a, base^a)) # initialize partial trace matrix
    ...
end # function partial_trace
```

You will also see that I am taking in some peculiar inputs. Because I possibly have a naive assumption that I can extend this algorithm to use something with more information that qubits, I added the base input which just gives us the dimension of those initial spaces. For qubits, this is just 2. I then have a, which defines the dimension we are taking the partial trace over as ⊗N-a (the last N-a qubits as I won't need the implementation the other way as far as I can tell). And finally ρ is the full density matrix. 

From here I do a loop over all indices in the partial trace matrix, then a loop over all "sub-diagonals," and add this to a sum inside our partial trace matrix. At the end I can return the partial trace which is *hopefully* correct (I know, I inspire confidence): 

```julia 
function partial_trace(a::Int, ρ::Matrix{ComplexF64}; base = 2)
    partialTrace = Matrix{Complex}(zeros(base^a, base^a)) # initialize partial trace matrix

    for index ∈ CartesianIndices(partialTrace) # loop through every index in partial trace (every basis combination for other space)
        for α ∈ 1:div(size(ρ)[1], size(partialTrace)[1]) # loop through the required dimension of the space we are taking a partial trace of
            partialTrace[index] += ρ[index[1]*α, index[2]*α] # add the corresponding density operator to the partial trace matrix
        end # for
    end # for

    partialTrace
end # function partial_trace
```

From here we want to diagonalize the partial trace operator. We can do that simply through eigenvalue decomposition. Doing this we end up with: UDU^† where we have eigenvalues in the diagonal matrix (D) and eigenvectors as columns in U (the dagger operator here is the complex conjugate transpose). Specifically, we are interested in the operator U. 

For philosophical reasons, the paper justifies dropping the two eigenvectors corresponding to the smallest eigenvalues. This is because we assume that paradigms like language have inherit structure and logic and we can relate removing the less important eigenvectors to removing sampling error. In other words, we want to remove sensitivity of the model so we can hopefully pull out factors and structure that have a larger influence over what we are modeling. 

By dropping these two eigenvectors, we end up with a 4 x 2 matrix. In code we can do this simply with: 

```julia 
U = eigvecs(partialTrace)[:, 3:4]
```

As Julia sorts our eigenvalues in order of importance for us, we can just take the later two. 

Okay, but we now have a matrix which can no longer act as an operator so we will lose dimensions applying it. What do we want to do with this? Well we actually do want to apply it, and every time we do apply one of these U matrices we will lose 1 qubit from our state vector. When we collapse all of our dimensions, we compose all of our Us together to create our final MPS, the desired vector. 

How do we apply this U? Well we reshape our state vector similar to how we did for the first step, just this time with the proper number of dimensions (2 qubits instead of 1 earlier): 

```julia
M_k = reshape(ψ_k, :, 4)
```

Then apply our U to this operator: 

```julia 
MU = M_k * U
```

And we reshape one final time to get the next state vector: 

```julia 
ψ_k = reshape(MU, :, 1)
```

All together these steps look like this: 

```julia 
for _ ∈ 1:N-2 # loop N-2 times 
    partialTrace = partial_trace(2, outer_product(ψ_k, ψ_k)) # calculate partial trace of the density matrix for ψ_k for ⊗N-2
    
    U = eigvecs(partialTrace)[:, 3:4] # find eigenvectors of partial trace and remove the two with the smallest eigenvalues

    M_k = reshape(ψ_k, :, 4) # reshape ψ_k into a map V^(⊗2) → V^(⊗N-2)

    MU = M_k * U # apply the U to the map 

    ψ_k = reshape(MU, :, 1) # reshape the map back into a vector

    push!(Umatrices, U) # push the U into the Umatrices vector
end # for
```

Now you will notice that this is looping N-2 times for an algorithm that is supposed to have N steps. Well the first step was applying the identity, so that is put outside the loop, and the last step is a little bit special. At the end of these, in total, N-2 steps, we have a ψ_k vector that is dimension 4. We reshape this into a linear mapping from a 2-dimensional space to another 2-dimensional space. We are going to use this as our final mapping in our MPS tensor (our output). This step is simple in code: 

```julia 
U_N = reshape(ψ_k, :, 2)
```

So here is where I am very unconfident that I am following the thesis (I was already a little bit unconfident that I was doing it right, this is the step I really doubt). This is the step of constructing the MPS vector. Essentially the entire mapping which we applied to the ψ vectors earlier, well we can think of our output vector as that mapping without the initial ψ vector we are applying everything to. This is a mapping itself which can be reshaped into the final output vector. 

As I am not entirely fluent tensor network diagrams, here is where my doubts really come in. But as I understand what is happening, we can think of this state as if we were applying quantum operators to various qubits. So if we wanted to apply the CNOT gate to the 2nd and 3rd qubit in a 5-qubit system, well we would use the opeartor: I2 ⊗ CNOT ⊗ I4. In this instance, we start with the identity operator on N-1 qubits. I am thinking about it this way both because this is how I can get a vector of the correct size but also because the U operators are applied to every qubit but the last qubit which we will think of incorporating as "pluggin in" our final U into that qubit. We then just apply U_k ⊗ Identity(2^(N-3-k+1)) for all k, at the end applying the 2x2 U_N to a m x 2 matrix. As a final step, we want to reshape this matrix such that instead of going "from the" last qubit to the others, it is just a vector holding N qubits. In total, this final bit of code is the following: 

```julia
# generate MPS 
MPS = Identity(2^(N-1)) # initialize MPS
for U ∈ Umatrices # loop through all Us 
    MPS = MPS * kron(U, Identity(2^(N-2-k))) 
end # for

MPS = MPS * U_N # apply the last U to the MPS

reshape(transpose(MPS), :, 1) # reshape the MPS into a vector
```

In full, the model, in code, looks like this: 

```julia
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
```

Our first step is to apply the first identity matrix as a formality. After that we start iterating N-2 times. On each iteration, we take the state vector at that step, which started by representing each possibility with the sqaure root of the probability in each index, and take the partial trace with respect to the last N-2 qubits of the density operator created by this state vector. This, in effect, finds the marginal probability of the first 2 qubits. We perform spectral decomposition on this matrix and throw out the eigenvectors corresponding the the smallest eigenvalues. This is done in an effort to reduce noise. These eigenvectors correspond to possible states of our subsystem (with corresponding eigenvalues being their probabilities of occuring). We reshape our state vector in order to apply the 4x2 matrix of eigenvectors to the first two qubits, in a way collapsing them into a combined state. We reshape this matrix back into a state vector, now with one less qubit, and continue the iteration. When the iteration is finished, we can simply define a final 2x2 operator, U_N, by reshaping the final state vector. And finally, using ideas from constructing quantum algorithms, we construct a map from N-1 qubits to 1 qubit from all of our Us (except the last one), and then apply the U_N matrix to the final qubit. After all of that, we reshape it back into a vector. We can think of this final vector as a MPS, matrix product state, which is the output of combining a bunch of matrices together. 

Now I am really not sure how good my model is. Anecdotally I might say "it almost looks like it works." For actual numbers, running on 5-bit strings giving it all even-bitstrings as values, it gave an average probability of 0.0073 to odd bit-strings and an average probability of 0.0298 to even bit-strings. The difference is maybe better seen in the medians with a median probability of 0.00298 given to odd and 0.0349 given to even. Maximums are also a little bit telling with odd maxing out at 0.0228 and even at 0.0666. For reference we were aiming for zero probability for an odd string and 0.0625 for even. Now obviously this is a terrible comparison, especially in comparison to the comparison in the thesis; however, I think it definitely shows that this model gave a much higher probability to even bit strings than odd bit strings which is really promising to it actually being correct, which I am still not sure on. Trying different lengths for the bitstrings showed a fairly inconsistent pattern though so I might have to lay on the side of I did something wrong. 