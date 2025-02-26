using MPI, Serialization

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

struct MPICallBack{T}
  comm::T
  rank::Int
  commsize::Int
end

(cb::MPICallBack)(A) = MPI.Barrier(cb.comm)
function (cb::MPICallBack)(A, x::Union{Matrix, Vector})
  MPI.Allreduce!(x, +, cb.comm)
  return x
end

function (cb::MPICallBack)(A, x) # works for typeof(x)<:Union{SparseMatrixCSC, BlockArray} at least
  s = IOBuffer()
  Serialization.serialize(s, x)
  s = take!(s)
  lens = MPI.Allgather(Int32(length(s)), cb.comm)
  g = MPI.Allgatherv(s, lens, cb.comm)
  d = deserialisechunks(g, vcat(0, cumsum(lens)), (a, b)->(a .+= b)) # double workaround
  view(x, :, :) .= view(d, :, :)
  return x
end

function (cb::MPICallBack)(A, x::Dict) # a work around
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

