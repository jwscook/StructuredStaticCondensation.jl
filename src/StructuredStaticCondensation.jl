module StructuredStaticCondensation

using MPI

using LinearAlgebra, Base.Threads

export SSCMatrix, factorise!

using StaticCondensation
import StaticCondensation: factorise!, free

include("mpiinclude.jl")

const DEFAULT_INPLACE = false

function allocatereducedlhs(A, totalcouplingblocksize, reducedcoupledindices, context)
  return zeros(eltype(A), totalcouplingblocksize, totalcouplingblocksize)
end
function allocatereducedlhs(A, totalcouplingblocksize, reducedcoupledindices,
    context::MPIContext)#{DistributedMemoryMPI})
  tmp = zeros(eltype(A), totalcouplingblocksize, totalcouplingblocksize)
  context = StaticCondensation.MPIContext(StaticCondensation.DistributedMemoryMPI(),
                                          context.comm, context.rank, context.size)
  return SCMatrix(tmp, reducedcoupledindices; context=context)
end
function allocatereducedlhs(A, totalcouplingblocksize, reducedcoupledindices,
    context::MPIContext{SharedMemoryMPI})

  tmp = zeros(eltype(A), totalcouplingblocksize, totalcouplingblocksize)
  sharedtmp, win = sharedmemorympimatrix(tmp, context)
  context = StaticCondensation.MPIContext(StaticCondensation.SharedMemoryMPI(win),
                                          context.comm, context.rank, context.size)
  return SCMatrix(sharedtmp, reducedcoupledindices; context=context)
end

struct SSCMatrix{T, M<:AbstractMatrix{T}, R, U, C} <: AbstractMatrix{T}
  A::M
  indices::Vector{UnitRange{Int}}
  nlocalblocks::Int
  ncouplingblocks::Int
  reducedlocalindices::Vector{UnitRange{Int}}
  reducedcoupledindices::Vector{UnitRange{Int}}
  reducedlhs::R
  enumeratelocalindices::U
  enumeratecouplingindices::U
  selectedlocalindices::Vector{Int}
  selectedcouplingindices::Vector{Int}
  alllocalindices::Vector{Int}
  allcouplingindices::Vector{Int}
  context::C
  function SSCMatrix(A::AbstractMatrix{T}, blockindices; context=SerialContext()
      ) where T
    n = size(A, 1)
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
    reducedlhs = allocatereducedlhs(A, totalcouplingblocksize, reducedcoupledindices, context)
    #reducedlhs = zeros(T, totalcouplingblocksize, totalcouplingblocksize)

    enumeratelocalindices = collect(zip(1:nlocalblocks, 1:2:length(blockindices), blockindices[1:2:end]))
    enumeratecouplingindices = collect(zip(1:ncouplingblocks, 2:2:length(blockindices), blockindices[2:2:end-1]))

    selectedlocalindices = collect(1:nlocalblocks)
    selectedcouplingindices = collect(1:ncouplingblocks)

    alllocalindices = reduce(vcat, collect.(li for (_, _, li) in enumeratelocalindices))
    allcouplingindices = reduce(vcat, collect.(li for (_, _, li) in enumeratecouplingindices))

    M = typeof(A)
    U = typeof(enumeratelocalindices)
    R = typeof(reducedlhs)
    C = typeof(context)

    output = new{T, M, R, U, C}(A, blockindices, nlocalblocks, ncouplingblocks,
      reducedlocalindices, reducedcoupledindices, reducedlhs,
      enumeratelocalindices, enumeratecouplingindices,
      selectedlocalindices, selectedcouplingindices,
      alllocalindices, allcouplingindices,
      context)
    distributeenumerations!(output)
    return output
  end
end

function free(A::SSCMatrix)
  free(A.context)
  free(A.reducedlhs)
  return nothing
end

function calculateindices(A, localblocksize, couplingblocksize)
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
  return indices
end
function SSCMatrix(A::AbstractMatrix{T}, localblocksize, couplingblocksize;
    context=SerialContext()) where T
  indices = calculateindices(A, localblocksize, couplingblocksize)
  return SSCMatrix(A, indices; context=context)
end
Base.size(A::SSCMatrix) = (size(A.A, 1), size(A.A, 2))
Base.size(A::SSCMatrix, i) = size(A.A, i)
islocalblock(i) = isodd(i)
iscouplingblock(i) = !islocalblock(i)

tile(A::SSCMatrix, i, j) = view(A.A, A.indices[i], A.indices[j])

function enumeratelocalindices(A::SSCMatrix)
  return A.enumeratelocalindices[A.selectedlocalindices]
end
function enumeratecouplingindices(A::SSCMatrix)
  return A.enumeratecouplingindices[A.selectedcouplingindices]
end

lutype(A::Matrix{T}) where T = LU{T, Matrix{T}, Vector{Int64}}

function calculatelocalfactors(A::SSCMatrix{T}; inplace=DEFAULT_INPLACE) where T
  localfactors = Dict{Int, lutype(A.A)}()
  for (c, i, li) in enumeratelocalindices(A) # parallelisable
    localfactors[i] = inplace ? lu!(tile(A, i, i)) : lu(tile(A, i, i))
  end
  return localfactors
end

couplingtype(A::AbstractMatrix) = typeof(A)

function calculatecouplings(A::SSCMatrix{T,M}, localfactors) where {T,M}
  couplings = Dict{Tuple{Int, Int}, couplingtype(A.A)}()
  # ith iterate needs localfactors i-1 and i+1
  @views for (c, i, li) in enumeratecouplingindices(A) # parallelisable
    if i - 1 >= 1
      couplings[(i-1, i)] = localfactors[i-1] \ tile(A, i-1, i)
    end
    if i + 1 <= length(A.indices)
      couplings[(i+1, i)] = localfactors[i+1] \ tile(A, i+1, i)
    end
  end
  return couplings
end

struct SSCMatrixFactorisation{T,M,R,U,V,W,X}
  A::SSCMatrix{T,M,R,U}
  lureducedlhs::V
  localfactors::W
  couplings::X
  reducedrhswork::Vector{T}
end

function free(A::SSCMatrixFactorisation)
  free(A.A)
  free(A.lureducedlhs)
  return nothing
end

function calculatelocalsolutions(F::SSCMatrixFactorisation{T,M,R}, b) where {T,M,R}
  d = Dict{Int, Matrix{T}}()
  @views for (c, i, li) in enumeratelocalindices(F.A) # parallelisable
    d[i] = F.localfactors[i] \ b[li, :]
  end
  return d
end

function assemblecoupledrhs(A::SSCMatrix{T}, B, localsolutions) where T
  m = size(A.reducedlhs, 1)
  n = size(B, 2)
  mn = n == 1 ? (m,) : (n, m)
  b = similar(B, mn...)
  fill!(b, 0)
  return assemblecoupledrhs!(b, A, B, localsolutions)
end
function assemblecoupledrhs!(b, A::SSCMatrix{T}, B, localsolutions) where T
  @views for (c, i, li) in enumeratecouplingindices(A) # parallelisable
    rows = A.reducedcoupledindices[c]
    b[rows, :] .= B[li, :]
    mul!(b[rows, :], tile(A, i, i-1), localsolutions[i - 1], -one(T), one(T))
    mul!(b[rows, :], tile(A, i, i+1), localsolutions[i + 1], -one(T), one(T))
  end
  return b
end

function assemblecoupledlhs!(A::SSCMatrix{T}, couplings;
    assignblocks=false, applycouplings=false) where T
  M = A.reducedlhs
  #  mul!(C, A, B, α, β) is C = A B α + C β.
  # ith iterate needs couplings (i±1, 1) and (i + 1, i + 2) and (i-1, i-2)
  @views for (c, i, li) in enumeratecouplingindices(A) # parallelisable
    rows = A.reducedcoupledindices[c]
    assignblocks && copyto!(view(M, rows, rows), tile(A, i, i))
    aim = tile(A, i, i-1)
    aip = tile(A, i, i+1)
    applycouplings && mul!(M[rows, rows], aim, couplings[(i - 1, i)], -one(T), one(T))
    applycouplings && mul!(M[rows, rows], aip, couplings[(i + 1, i)], -one(T), one(T))
    if c + 1 <= A.ncouplingblocks
      right = A.reducedcoupledindices[c + 1]
      assignblocks && copyto!(view(M, rows, right), tile(A, i, i + 2))
      applycouplings && mul!(M[rows, right], aip, couplings[(i + 1, i + 2)], -one(T), one(T))
    end
    if c - 1 >= 1
      left = A.reducedcoupledindices[c - 1]
      assignblocks && copyto!(view(M, rows, left), tile(A, i, i - 2))
      applycouplings && mul!(M[rows, left], aim, couplings[(i - 1, i - 2)], -one(T), one(T))
    end
  end
  return M
end

function coupledx!(x, Ac, bc, allcouplingindices, _)
  xc = view(x, allcouplingindices, :)
  #copyto!(xc, bc); ldiv!(Ac, xc)
  ldiv!(xc, Ac, bc)
  return x
end
function coupledx!(x, Ac::LU{T, Matrix{T}}, bc, allcouplingindices, rhswork) where T
  xc = view(x, allcouplingindices, :)
  for j in 1:size(bc, 2)
    copyto!(rhswork, view(bc, :, j))
    LinearAlgebra._apply_ipiv_rows!(Ac, rhswork)
    LinearAlgebra.LAPACK.trtrs!('L', 'N', 'U', Ac.factors, rhswork)
    LinearAlgebra.LAPACK.trtrs!('U', 'N', 'N', Ac.factors, rhswork)
    copyto!(view(xc, :, j), rhswork)
  end
  return x
end

function coupledx!(x, F::SSCMatrixFactorisation{T}, bc) where {T}
  Ac = F.lureducedlhs
  return coupledx!(x, Ac, bc, F.A.allcouplingindices, F.reducedrhswork)
end

function localx!(x, cx, A::SSCMatrix{T}, localsolutions, couplings) where T
  @inbounds for (c, i, li) in enumeratelocalindices(A) # parallelisable
    rows = A.reducedlocalindices[c]
    copyto!(view(x, li, :), localsolutions[i])
    for j in (i + 1, i - 1)
      0 < j <= length(A.indices) || continue
      #  mul!(C, A, B, α, β) is C = A B α + C β.
      @views mul!(x[li, :], couplings[(i, j)], cx[A.indices[j], :], -one(T), one(T))
    end
  end
  return x
end

function distributeenumerations!(A::SSCMatrix{T,U,V,W,<:SerialContext}
    ) where {T,U,V,W}
  return nothing
end
function distributeenumerations!(A::SSCMatrix{T,U,V,W,<:MPIContext}
    ) where {T,U,V,W}
  rank = A.context.rank
  commsize = A.context.size

  mylocalindices = distributeenumerations!(A.nlocalblocks, rank, commsize)
  resize!(A.selectedlocalindices, length(mylocalindices))
  A.selectedlocalindices .= mylocalindices

  mycouplingindices = distributeenumerations!(A.ncouplingblocks, rank, commsize)
  resize!(A.selectedcouplingindices, length(mycouplingindices))
  A.selectedcouplingindices .= mycouplingindices
end

function distributeenumerations!(itlength::Int, rank, commsize)
  @assert 0 <= rank < commsize
  # first get the indices we want to keep
  indices = rank+1:commsize:itlength
  # then delete all other entires using filter
  output = collect(1:itlength)
  deleteat!(output,
            filter!(i->!in(i, indices), collect(1:itlength)))
  return output
end

factorise!(A) = lu!(A)

function factorise!(A::SSCMatrix; inplace=DEFAULT_INPLACE)
  A.context(A)
  assemblecoupledlhs!(A, nothing; assignblocks=true)
  A.context(A)
  localfactors = calculatelocalfactors(A; inplace=inplace)
  # share localfactors so that coupling iterate i has i+1 and i-1
  A.context(A, localfactors)
  couplings = calculatecouplings(A, localfactors)
  # share couplines so ith coupling iterate has (i±1, i), (i+1, i+2),(i-1, i-2)
  A.context(A, couplings)
  assemblecoupledlhs!(A, couplings; applycouplings=true)
  A.context(A, A.reducedlhs)

  lureducedlhs = factorise!(A.reducedlhs)
  reducedrhswork = zeros(eltype(A), size(A.reducedlhs, 1))

  return SSCMatrixFactorisation(A, lureducedlhs, localfactors, couplings, reducedrhswork)
end

function LinearAlgebra.ldiv!(A::SSCMatrixFactorisation{T}, b) where {T}
  x = zeros(T, size(b))
  return ldiv!(x, A, b)
end

function LinearAlgebra.ldiv!(x, F::SSCMatrixFactorisation{T}, b) where {T}

  F.A.context(F)
  localsolutions = calculatelocalsolutions(F, b)
  F.A.context(F, localsolutions)

  bc = assemblecoupledrhs(F.A, b, localsolutions)
  F.A.context(F, bc)

  cx = x
  cx = coupledx!(cx, F, bc)

  lx = cx
  lx = localx!(lx, cx, F.A, localsolutions, F.couplings)
  F.A.context(F, view(lx, F.A.alllocalindices, :))

  return x
end

function LinearAlgebra.ldiv!(A::SSCMatrix{T}, b; inplace=DEFAULT_INPLACE) where T
  F = factorise!(A; inplace=inplace)
  return ldiv!(F, b)
end

end # module StructuredSSC
