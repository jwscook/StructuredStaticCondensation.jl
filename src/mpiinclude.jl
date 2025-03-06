using MPI, Serialization

abstract type AbstractContext end

struct SerialContext <: AbstractContext end
(cb::SerialContext)(args...) = nothing

returnfunc(x::AbstractArray{<:Number}, op) = x
returnfunc(x::AbstractArray, op) = reduce(op, x)
function returnfunc(x::Vector{<:AbstractArray}, op)
  if length(x) == 1
    return x[1]
  else
    for i in 2:length(x)
      op(view(x[1], :, :), view(x[i], :, :))
    end
    return x[1]
  end
end

function deserialisechunks(x, lens, op)
  y = [deserialize(IOBuffer(x[lens[i]+1:lens[i+1]])) for i in 1:length(lens)-1]
  return returnfunc(y, op) #reduce(op, y)
end

abstract type AbstractMPI end
struct DistributedMemoryMPI <: AbstractMPI end
struct SharedMemoryMPI <: AbstractMPI
  win::MPI.Win
end

struct MPIContext{T<:AbstractMPI, C} <: AbstractContext
  mpitype::T
  comm::C
  rank::Int
  size::Int
  function MPIContext(mpitype::T, comm::C, rank::Int, size::Int) where {T<:AbstractMPI, C}
    @assert 0 <= rank < size
    return new{T, C}(mpitype, comm, rank, size)
  end
end

(cb::MPIContext)(A) = MPI.Barrier(cb.comm)
function (cb::MPIContext{<:DistributedMemoryMPI})(A, x::Union{Matrix, Vector})
  MPI.Allreduce!(x, +, cb.comm)
  return x
end
function (cb::MPIContext{<:SharedMemoryMPI})(A, x::Union{Matrix, Vector})
  MPI.Allreduce!(x, +, cb.comm)
  return x
end
function (cb::MPIContext{<:SharedMemoryMPI})(A,
    x::SCMatrix{T,M,<:StaticCondensation.MPIContext{<:StaticCondensation.SharedMemoryMPI}}) where {T,M}
  MPI.Barrier(cb.comm)
  return x
end

function (cb::MPIContext)(A, x) # works for typeof(x)<:Union{SparseMatrixCSC, BlockArray} at least
  s = IOBuffer()
  Serialization.serialize(s, x)
  s = take!(s)
  lens = MPI.Allgather(Int32(length(s)), cb.comm)
  g = MPI.Allgatherv(s, lens, cb.comm)
  d = deserialisechunks(g, vcat(0, cumsum(lens)), (a, b)->(a .+= b)) # double workaround work Allreduce
  view(x, :, :) .= view(d, :, :)
  return x
end

function (cb::MPIContext)(A, x::Dict) # a work around
  # this could be better - send key-val pairs directly to rank that needs them
  s = IOBuffer()
  Serialization.serialize(s, x)
  s = take!(s)
  lens = MPI.Allgather(Int32(length(s)), cb.comm)
  g = MPI.Allgatherv(s, lens, cb.comm)
  d = deserialisechunks(g, vcat(0, cumsum(lens)), merge) # double workaround
  merge!(x, d)
  MPI.Barrier(cb.comm)
  return x
end

sharedmemorympimatrix(tmparray, con::SerialContext) = tmparray
function sharedmemorympimatrix(tmparray, con::MPIContext)
  return sharedmemorympimatrix(tmparray, con.comm, con.rank)
end
function sharedmemorympimatrix(tmparray, sharedcomm, sharedrank)
  dimslocal = sharedrank == 0 ? size(tmparray) : (0, 0)
  win, arrayptr = MPI.Win_allocate_shared(Array{Float64}, prod(dimslocal), sharedcomm)
  MPI.Barrier(sharedcomm)
  A = MPI.Win_shared_query(Array{Float64}, prod(size(tmparray)), win; rank=0)
  A = reshape(A, size(tmparray))
  sharedrank == 0 && (A .= tmparray)
  MPI.Barrier(sharedcomm)
  return A, win
end


