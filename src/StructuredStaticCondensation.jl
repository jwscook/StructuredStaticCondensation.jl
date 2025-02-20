module StructuredStaticCondensation

using LinearAlgebra

export SSCMatrix, factorise!

struct SSCMatrix{T, M<:AbstractMatrix{T}} <: AbstractMatrix{T}
  A::M
  indices::Vector{UnitRange{Int}}
  nlocalblocks::Int
  ncouplingblocks::Int
  reducedlocalindices::Vector{UnitRange{Int}}
  reducedcoupledindices::Vector{UnitRange{Int}}
  reducedlhs::M
end

function SSCMatrix(A::AbstractMatrix{T}, blockindices) where T
  n = size(A, 1)
  @assert isodd(length(blockindices))
  ncouplingblocks = (length(blockindices) - 1) ÷ 2
  nlocalblocks = ncouplingblocks + 1

  reducedlocalindices = Vector{UnitRange{Int}}()
  push!(reducedlocalindices, blockindices[1])
  for (c, i) in enumerate(3:2:length(blockindices))
    inds = (blockindices[i] .- blockindices[i][1] .+ 1) .+ reducedlocalindices[c][end]
    push!(reducedlocalindices, inds) 
  end
  reducedcoupledindices = Vector{UnitRange{Int}}()
  push!(reducedcoupledindices, blockindices[2] .- blockindices[2][1] .+ 1)
  for (c, i) in enumerate(4:2:length(blockindices))
    inds = (blockindices[i] .- blockindices[i][1] .+ 1) .+ reducedcoupledindices[c][end]
    push!(reducedcoupledindices, inds) 
  end

  totalcouplingblocksize = sum(length(i) for i in reducedcoupledindices)
  # allocate reduced lhs
  reducedlhs = similar(A, totalcouplingblocksize, totalcouplingblocksize)
  fill!(reducedlhs, 0)

  return SSCMatrix(A, blockindices, nlocalblocks, ncouplingblocks, reducedlocalindices, reducedcoupledindices, reducedlhs)
end

function SSCMatrix(A::AbstractMatrix{T}, localblocksize, couplingblocksize) where T
  n = size(A, 1)
  ncouplingblocks = (n - localblocksize) ÷ (localblocksize + couplingblocksize)
  nlocalblocks = ncouplingblocks + 1

  indices = Vector{UnitRange{Int}}()
  a = 1
  for i = 1:nlocalblocks-1
    inds = a:a + localblocksize - 1
    push!(indices, inds)
    a = a + localblocksize
    inds = a:a + couplingblocksize - 1
    push!(indices, inds)
    a = a + couplingblocksize
  end
  push!(indices, a:a + localblocksize - 1)
  @assert indices[end][end] == size(A, 1) == size(A, 2)
  return SSCMatrix(A, indices)
end
Base.size(A::SSCMatrix) = (size(A.A, 1), size(A.A, 2))
Base.size(A::SSCMatrix, i) = size(A.A, i)
islocalblock(i) = isodd(i)
iscouplingblock(i) = !islocalblock(i)

function enumeratelocalindices(A::SSCMatrix)
  return zip(1:A.nlocalblocks, 1:2:length(A.indices), A.indices[1:2:end])
end
function enumeratecouplingindices(A::SSCMatrix)
  return zip(1:A.ncouplingblocks, 2:2:length(A.indices), A.indices[2:2:end-1])
end

function factoriselocals(A::SSCMatrix{T}) where T
  localfact = lu!(view(A.A, A.indices[1], A.indices[1])) # can't use a view
  d = Dict{Int, typeof(localfact)}(1=>localfact)
  #for (i_1, li) in enumerate(A.indices[2:end]) # parallelisable
  for (c, i, li) in collect(enumeratelocalindices(A))[2:end]# parallelisable
    d[i] = lu!(view(A.A, li, li)) # can't use a view
  end
  return d
end

function calculatecouplings(A::SSCMatrix{T,M}, localfactors) where {T,M}
  d = Dict{Tuple{Int, Int}, M}()
  for (c, i, li) in enumeratecouplingindices(A) # parallelisable
    if i - 1 >= 1
      lim = A.indices[i-1]
      d[(i-1, i)] = localfactors[i-1] \ A.A[lim, li]
    end
    if i + 1 <= length(A.indices)
      lip = A.indices[i+1]
      d[(i+1, i)] = localfactors[i+1] \ A.A[lip, li]
    end
  end
  return d
end

struct SSCMatrixFactorisation{T,M,U,V}
  A::SSCMatrix{T,M}
  localfactors::U
  couplings::V
end

function solvelocalparts(F::SSCMatrixFactorisation{T,M}, b) where {T,M}
  d = Dict{Int, M}()
  for (c, i, li) in enumeratelocalindices(F.A) # parallelisable
    d[i] = F.localfactors[i] \ b[li, :]
  end
  return d
end

function couplingblockindices(A::SSCMatrix, i)
  @assert i > 1
  return A.indices[i] .- A.indices[i-1][1] .+ 1
end

function localblockindices(A::SSCMatrix, i)
  @assert i > 1
  return A.indices[i] .- A.indices[i-1][1] .+ 1
end

function assemblecoupledrhs(A::SSCMatrix, B, localsolutions, couplings)
  b = similar(A.reducedlhs, size(A.reducedlhs, 1), size(B, 2))
  return assemblecoupledrhs!(b, A, B, localsolutions, couplings)
end
function assemblecoupledrhs!(b, A::SSCMatrix, B, localsolutions, couplings)
  @views for (c, i, li) in enumeratecouplingindices(A) # parallelisable
    rows = A.reducedcoupledindices[c]
    b[rows, :] .= B[li, :]
    b[rows, :] .-= A.A[li, A.indices[i-1]] * localsolutions[i-1]
    b[rows, :] .-= A.A[li, A.indices[i+1]] * localsolutions[i+1]
  end
  return b
end

function assemblecoupledlhs!(A::SSCMatrix, couplings; assignblocks=false, applycouplings=false)
  M = A.reducedlhs
  @views for (c, i, li) in enumeratecouplingindices(A) # parallelisable
    rows = A.reducedcoupledindices[c]
    assignblocks && (M[rows, rows] .= A.A[li, li])
    aim = A.A[li, A.indices[i-1]]
    aip = A.A[li, A.indices[i+1]]
    applycouplings && (M[rows, rows] .-= aim * couplings[(i - 1, i)])
    applycouplings && (M[rows, rows] .-= aip * couplings[(i + 1, i)])
    if c + 1 <= A.ncouplingblocks
      right = A.reducedcoupledindices[c + 1]
      assignblocks && (M[rows, right] .= A.A[li, A.indices[i + 2]])
      applycouplings && (M[rows, right] .-= aip * couplings[(i + 1, i + 2)])
    end
    if c - 1 >= 1
      left = A.reducedcoupledindices[c - 1]
      assignblocks && (M[rows, left] .= A.A[li, A.indices[i - 2]])
      applycouplings && (M[rows, left] .-= aim * couplings[(i - 1, i - 2)])
    end
  end
  return M
end

function coupledx!(x, A::SSCMatrix{T}, b, localsolutions, couplings) where {T}
  Ac = assemblecoupledlhs!(A, couplings; applycouplings=true)
  bc = assemblecoupledrhs(A, b, localsolutions, couplings)
  xc = Ac \ bc
  for (c, i, ind) in enumeratecouplingindices(A)
    x[ind, :] .= xc[A.reducedcoupledindices[c], :]
  end
  return x
end

function localx!(x, A::SSCMatrix{T}, localsolutions, couplings) where T
  for (c, i, li) in enumeratelocalindices(A) # parallelisable
    rows = A.reducedlocalindices[c]
    x[li, :] .= localsolutions[i]
    for j in (i + 1, i - 1)
      0 < j <= length(A.indices) || continue
      x[li, :] .-= couplings[(i, j)] * x[A.indices[j], :]
    end
  end
  return x
end

function factorise!(A::SSCMatrix)
  assemblecoupledlhs!(A, nothing; assignblocks=true)
  localfactors = factoriselocals(A)
  couplings = calculatecouplings(A, localfactors)
  return SSCMatrixFactorisation(A, localfactors, couplings)
end

function LinearAlgebra.ldiv!(A::SSCMatrixFactorisation{T}, b) where T

  localsolutions = solvelocalparts(A, b)

  x = zeros(T, size(b))
  x = coupledx!(x, A.A, b, localsolutions, A.couplings)
  x = localx!(x, A.A, localsolutions, A.couplings)
  return x
end

function LinearAlgebra.ldiv!(A::SSCMatrix{T}, b) where T
  F = factorise!(A)

  localsolutions = solvelocalparts(F, b)

  x = zeros(T, size(b))
  x = coupledx!(x, F.A, b, localsolutions, F.couplings)
  x = localx!(x, F.A, localsolutions, F.couplings)
  return x
end

end # module StructuredSSC
