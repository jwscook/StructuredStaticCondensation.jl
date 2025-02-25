using MPI
MPI.Init()
using Serialization

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

using StructuredStaticCondensation

using Random, Test, LinearAlgebra

@testset "SSCCondensation" begin
  comm = MPI.COMM_WORLD
  commsize = MPI.Comm_size(comm)
  rank = MPI.Comm_rank(comm)
  Random.seed!(rank)

  function dotest(tmp, L, C)
    A = MPI.bcast(tmp, 0, comm)
    b = MPI.bcast(rand(size(A, 1)), 0, comm)
    x = A \ b

    SCM = SSCMatrix(A, L, C)
    StructuredStaticCondensation.distributeenumerations!(SCM, rank, commsize)
    MPI.Barrier(comm)

    lens = MPI.Allgather(Int32(length(SCM.selectedcouplingindices)), comm)
    allcouplingindices = MPI.Allgatherv(SCM.selectedcouplingindices, lens, comm)
    allcouplingindices = sort([allcouplingindices...])
    @test allcouplingindices == 1:SCM.ncouplingblocks

    lens = MPI.Allgather(Int32(length(SCM.selectedlocalindices)), comm)
    alllocalindices = MPI.Allgatherv(SCM.selectedlocalindices, lens, comm)
    alllocalindices = sort([alllocalindices...])
    @test alllocalindices == 1:SCM.nlocalblocks

    SCMf = factorise!(deepcopy(SCM);
      callback=MPICallBack(comm, rank, commsize), inplace=false)
    for (c, i, li) in SCM.enumeratelocalindices
      lf = SCMf.localfactors[i]
      @test lf == lu(A[li, li])
    end
    for (c, i, li) in SCM.enumeratecouplingindices
      cp = SCMf.couplings[i-1, i]
      @test cp == SCMf.localfactors[i-1] \ (A[SCM.indices[i-1], li])
      cp = SCMf.couplings[i+1, i]
      @test cp == SCMf.localfactors[i+1] \ (A[SCM.indices[i+1], li])
    end
    @test ldiv!(SCM, b; callback=MPICallBack(comm, rank, commsize), inplace=false) ≈ x
  end

  for (L, C) in ((3, 2), (4, 2), (16, 4), (5, 7), (128, 64), (256, 128), (1024, 512))
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
