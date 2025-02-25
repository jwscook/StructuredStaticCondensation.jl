module StructuredStaticCondensation

using LinearAlgebra

export SSCMatrix, factorise!

const DEFAULT_INPLACE = true

struct SSCMatrix{T, M<:AbstractMatrix{T}, U} <: AbstractMatrix{T}
  A::M
  indices::Vector{UnitRange{Int}}
  nlocalblocks::Int
  ncouplingblocks::Int
  reducedlocalindices::Vector{UnitRange{Int}}
  reducedcoupledindices::Vector{UnitRange{Int}}
  reducedlhs::M
  enumeratelocalindices::U
  enumeratecouplingindices::U
  selectedlocalindices::Vector{Int}
  selectedcouplingindices::Vector{Int}
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

    enumeratelocalindices = collect(zip(1:nlocalblocks, 1:2:length(blockindices), blockindices[1:2:end]))
    enumeratecouplingindices = collect(zip(1:ncouplingblocks, 2:2:length(blockindices), blockindices[2:2:end-1]))

    selectedlocalindices = collect(1:nlocalblocks)
    selectedcouplingindices = collect(1:ncouplingblocks)

    M = typeof(A)
    U = typeof(enumeratelocalindices)

    return new{T, M, U}(A, blockindices, nlocalblocks, ncouplingblocks,
      reducedlocalindices, reducedcoupledindices, reducedlhs,
      enumeratelocalindices, enumeratecouplingindices,
      selectedlocalindices, selectedcouplingindices)
  end
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
  return A.enumeratelocalindices[A.selectedlocalindices]
end
function enumeratecouplingindices(A::SSCMatrix)
  return A.enumeratecouplingindices[A.selectedcouplingindices]
end

lutype(A::Matrix{T}) where T = LU{T, Matrix{T}, Vector{Int64}}

function calculatelocalfactors(A::SSCMatrix{T}; inplace=DEFAULT_INPLACE) where T
  d = Dict{Int, lutype(A.A)}()
  for (c, i, li) in enumeratelocalindices(A) # parallelisable
    d[i] = inplace ? lu!(view(A.A, li, li)) : lu(view(A.A, li, li))
#    @assert all(isfinite, d[i].L) # remove after debugged
#    @assert all(isfinite, d[i].U) # remove after debugged
  end
  return d
end

function calculatecouplings(A::SSCMatrix{T,M}, localfactors) where {T,M}
  d = Dict{Tuple{Int, Int}, M}()
  for (c, i, li) in enumeratecouplingindices(A) # parallelisable
    if i - 1 >= 1
      lim = A.indices[i-1]
      d[(i-1, i)] = localfactors[i-1] \ A.A[lim, li]
#      @assert all(isfinite, d[(i-1, i)]) # remove after debugged
    end
    if i + 1 <= length(A.indices)
      lip = A.indices[i+1]
      d[(i+1, i)] = localfactors[i+1] \ A.A[lip, li]
#      @assert all(isfinite, d[(i+1, i)]) # remove after debugged
    end
  end
  return d
end

struct SSCMatrixFactorisation{T,M,U,V}
  A::SSCMatrix{T,M}
  localfactors::U
  couplings::V
end

function calculatelocalsolutions(F::SSCMatrixFactorisation{T,M}, b) where {T,M}
  d = Dict{Int, M}()
  for (c, i, li) in enumeratelocalindices(F.A) # parallelisable
    d[i] = F.localfactors[i] \ b[li, :]
#    @assert all(isfinite, d[i]) # remove after debugged
  end
  return d
end

function assemblecoupledrhs(A::SSCMatrix, B, localsolutions)
  b = similar(A.reducedlhs, size(A.reducedlhs, 1), size(B, 2))
  fill!(b, 0)
  return assemblecoupledrhs!(b, A, B, localsolutions)
end
function assemblecoupledrhs!(b, A::SSCMatrix, B, localsolutions)
  @views for (c, i, li) in enumeratecouplingindices(A) # parallelisable
    rows = A.reducedcoupledindices[c]
    b[rows, :] .= B[li, :]
    b[rows, :] .-= A.A[li, A.indices[i-1]] * localsolutions[i-1]
    b[rows, :] .-= A.A[li, A.indices[i+1]] * localsolutions[i+1]
  end
  return b
end

function assemblecoupledlhs!(A::SSCMatrix, couplings;
    assignblocks=false, applycouplings=false)
  M = A.reducedlhs
  @views for (c, i, li) in enumeratecouplingindices(A) # parallelisable
    rows = A.reducedcoupledindices[c]
    assignblocks && (M[rows, rows] .= A.A[li, li])
    aim = view(A.A, li, A.indices[i-1])
    aip = view(A.A, li, A.indices[i+1])
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

function coupledx!(x, A::SSCMatrix{T}, bc; callback=defaultcallback) where {T}
  Ac = A.reducedlhs
  xc = Ac \ bc
#  @assert all(isfinite, xc) # remove after debugged
  for (c, i, ind) in enumeratecouplingindices(A)
    x[ind, :] .= xc[A.reducedcoupledindices[c], :]
  end
  return x
end

function localx!(x, cx, A::SSCMatrix{T}, localsolutions, couplings) where T
  for (c, i, li) in enumeratelocalindices(A) # parallelisable
    rows = A.reducedlocalindices[c]
    x[li, :] .= localsolutions[i]
    for j in (i + 1, i - 1)
      0 < j <= length(A.indices) || continue
      x[li, :] .-= couplings[(i, j)] * cx[A.indices[j], :]
    end
  end
#  @assert all(isfinite, x) # remove after debugged
  return x
end

defaultcallback(args...) = nothing

function distributeenumerations!(A::SSCMatrix, rank, commsize)
  @assert 0 <= rank < commsize
  # first get the indices we want to keep
  indices = rank+1:commsize:length(A.enumeratelocalindices)
  # then delete all other entires using filter
  deleteat!(A.selectedlocalindices,
    filter!(i->!in(i, indices), collect(eachindex(A.selectedlocalindices))))
  indices = rank+1:commsize:length(A.selectedcouplingindices)
  deleteat!(A.selectedcouplingindices,
    filter!(i->!in(i, indices), collect(eachindex(A.selectedcouplingindices))))
end

function factorise!(A::SSCMatrix; callback=defaultcallback, inplace=DEFAULT_INPLACE)
  callback(A)
  assemblecoupledlhs!(A, nothing; assignblocks=true)
  callback(A)
  localfactors = calculatelocalfactors(A; inplace=inplace)
  callback(A, localfactors)
  couplings = calculatecouplings(A, localfactors)
  callback(A, couplings)
  assemblecoupledlhs!(A, couplings; applycouplings=true)
  callback(A, A.reducedlhs)
  return SSCMatrixFactorisation(A, localfactors, couplings)
end

# inplace won't do anything, but this is needed to keep the API consistent
function LinearAlgebra.ldiv!(A::SSCMatrixFactorisation{T}, b;
    callback=defaultcallback, inplace=DEFAULT_INPLACE) where {T}

  callback(A)
  localsolutions = calculatelocalsolutions(A, b)
  callback(A, localsolutions)

  # build the solution out of two vectors so that the second callback
  # doesn't double count values during its MPI reduction
  # this could be better, if 
  bc = assemblecoupledrhs(A.A, b, localsolutions)
  callback(A, bc)

  cx = zeros(T, size(b))
  cx = coupledx!(cx, A.A, bc; callback=callback)
  callback(A,cx)

  lx = zeros(T, size(b))
  lx = localx!(lx, cx, A.A, localsolutions, A.couplings)
  callback(A, lx)

  return cx .+ lx
end

function LinearAlgebra.ldiv!(A::SSCMatrix{T}, b;
    callback=defaultcallback, inplace=DEFAULT_INPLACE) where T
  F = factorise!(A; callback=callback, inplace=inplace)
  return ldiv!(F, b; callback=callback)
end

end # module StructuredSSC
