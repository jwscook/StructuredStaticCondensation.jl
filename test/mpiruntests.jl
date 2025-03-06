using MPI
MPI.Init()
using MPI

include("matrixbuilder.jl")

using StructuredStaticCondensation

using Random, Test, LinearAlgebra, SparseArrays

@testset "SSCCondensation" begin
  comm = MPI.COMM_WORLD
  commsize = MPI.Comm_size(comm)
  rank = MPI.Comm_rank(comm)
  Random.seed!(rank)

  function dotest(L, C, nblocks)
    A, x, b = buildmatrix(L, C, nblocks)
    A = MPI.bcast(A, 0, comm)
    b = MPI.bcast(b, 0, comm)
    x = MPI.bcast(x, 0, comm)

    context = StructuredStaticCondensation.MPIContext(
      StructuredStaticCondensation.DistributedMemoryMPI(), comm, rank, commsize)
    SCM = SSCMatrix(A, L, C; context=context)
    MPI.Barrier(comm)

    lens = MPI.Allgather(Int32(length(SCM.selectedcouplingindices)), comm)
    allcouplingindices = MPI.Allgatherv(SCM.selectedcouplingindices, lens, comm)
    allcouplingindices = sort([allcouplingindices...])
    @test allcouplingindices == 1:SCM.ncouplingblocks

    lens = MPI.Allgather(Int32(length(SCM.selectedlocalindices)), comm)
    alllocalindices = MPI.Allgatherv(SCM.selectedlocalindices, lens, comm)
    alllocalindices = sort([alllocalindices...])
    @test alllocalindices == 1:SCM.nlocalblocks
    SCMf = factorise!(deepcopy(SCM); inplace=true)
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
    bcopy = deepcopy(b)
    @test ldiv!(SCM, b; inplace=false) ≈ x
    @test ldiv!(SCMf, bcopy) ≈ x
  end

  for (L, C) in ((16, 4), (128, 64), (256, 128), (1024, 512))
    for nblocks in (3, 5, 7, 9)
      dotest(L, C, nblocks)
    end
  end
end

MPI.Finalize()
