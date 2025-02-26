using MPI, Serialization

function deserialisechunks(x, lens, op=merge)
  y = [deserialize(IOBuffer(x[lens[i]+1:lens[i+1]])) for i in 1:length(lens)-1]
  return reduce(op, y)
end

struct MPICallBack{T}
  comm::T
  rank::Int
  commsize::Int
end

(cb::MPICallBack)(A) = MPI.Barrier(cb.comm)
function (cb::MPICallBack)(A, x::AbstractArray)
  MPI.Allreduce!(x, +, cb.comm)
  return x
end

function (cb::MPICallBack)(A, x::Dict) # a work around
  # this could be better - send key-val pairs directly to rank that needs them
  s = IOBuffer()
  Serialization.serialize(s, x)
  s = take!(s)
  lens = MPI.Allgather(Int32(length(s)), cb.comm)
  g = MPI.Allgatherv(s, lens, cb.comm)
  d = deserialisechunks(g, vcat(0, cumsum(lens))) # double workaround
  merge!(x, d)
  MPI.Barrier(cb.comm)
  return x
end

