using MPI
MPI.Init()
using Serialization

struct MPICallBack{T}
  comm::T
end

(cb::MPICallBack)(x::AbstractArray) = (x .= MPI.Allreduce!(x, +, cb.comm))

function (cb::MPICallBack)(x::Dict) # a work around
  s = IOBuffer()
  Serialization.serialize(s, x)
  s = take!(s)
#  s = StructuredStaticCondensation.serialise(x)
  lens = MPI.Allgather(Int32(length(s)), cb.comm)
  g = MPI.Allgatherv(s, lens, cb.comm)
  d = Serialization.deserialize(IOBuffer(g))
  #len = minimum(filter(!iszero, lens)) # double workaround
  #d = StructuredStaticCondensation.deserialise(g, len) # double workaround
  #merge!(x, d)
  mergewith((a, b)->a, x, d)
  MPI.Barrier(cb.comm)
  @show length(x)
  @show keys(x)
  @show values(x)
  return x
end

using StructuredStaticCondensation

function makesharedmatrix(sharedrank, sharedcomm, tmp)
  dimslocal = sharedrank == 0 ? size(tmp) : (0, 0)
  win, arrayptr = MPI.Win_allocate_shared(Array{Float64}, prod(dimslocal), sharedcomm)
  MPI.Barrier(sharedcomm)
  A = MPI.Win_shared_query(Array{Float64}, prod(size(tmp)), win; rank=0)
  A = reshape(A, size(tmp))
  MPI.Barrier(sharedcomm)
  sharedrank == 0 && (A .= tmp)
  return A
end

using Random, Test, LinearAlgebra
Random.seed!(0)

@testset "SSCCondensation" begin
  comm = MPI.COMM_WORLD
  commsize = MPI.Comm_size(comm)
  rank = MPI.Comm_rank(comm)
  Random.seed!(rank)
  color = rank < commsize ÷ 2 ? 0 : 1 # split into two shared memory mpi communicators
  sharedcomm = MPI.Comm_split(comm, color, rank)
  sharedrank = MPI.Comm_rank(sharedcomm)
  sharedsize = MPI.Comm_size(sharedcomm)

  function dotest(tmp, L, C)
    A = makesharedmatrix(sharedrank, sharedcomm, tmp)
    (x,b) = if sharedrank == 0
      b = rand(size(A, 1))
      x = tmp \ b
      (x, b)
    else
      (nothing, nothing)
    end
    x = MPI.bcast(x, 0, sharedcomm)
    b = MPI.bcast(b, 0, sharedcomm)

    SCM = SSCMatrix(A, L, C)
#    StructuredStaticCondensation.distributeenumerations!(SCM, sharedrank, sharedsize)
    @test ldiv!(SCM, b; callback=MPICallBack(sharedcomm)) ≈ x
  end

  for (L, C) in ((3, 2), )#(3, 2), (4, 2), (16, 4), (5, 7), (128, 64), (256, 128), (1024, 512))
    A11, A33, A55, A77 = (rand(L, L) for _ in 1:4)
    t12, t32, t34, t54, t56, t76 = (rand(L, C) for _ in 1:6)
    s21, s23, s43, s45, s65, s67 = (rand(C, L) for _ in 1:6)
    a22, a44, a66, a24, a42, a46, a64 = (rand(C, C) for _ in 1:7)
    zLL = zeros(L, L)
    zCL = zeros(C, L)
    zLC = zeros(L, C)
    zCC = zeros(C, C)
   
    tmp = [A11 t12 zLL;
           s21 a22 s23;
           zLL t32 A33]
    dotest(tmp, L, C)
    
    tmp = [A11 t12 zLL zLC zLL;
           s21 a22 s23 a24 zCL;
           zLL t32 A33 t34 zLL;
           zCL a42 s43 a44 s45;
           zLL zLC zLL t54 A55]
    dotest(tmp, L, C)
    
    tmp = [A11 t12 zLL zLC zLL zLC zLL;
           s21 a22 s23 a24 zCL zCC zCL;
           zLL t32 A33 t34 zLL zLC zLL;
           zCL a42 s43 a44 s45 a46 zCL;
           zLL zLC zLL t54 A55 t56 zLL;
           zCL zCC zCL a64 s65 a66 s67;
           zLL zLC zLL zLC zLL t76 A77]
    dotest(tmp, L, C)
  end
end

MPI.Finalize()
