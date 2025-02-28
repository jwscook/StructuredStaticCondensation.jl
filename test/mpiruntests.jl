using MPI
MPI.Init()

include("mpiinclude.jl")
include("matrixbuilder.jl")

using StructuredStaticCondensation

using Random, Test, LinearAlgebra, SparseArrays

@testset "SSCCondensation" begin
  comm = MPI.COMM_WORLD
  commsize = MPI.Comm_size(comm)
  rank = MPI.Comm_rank(comm)
  Random.seed!(rank)

  function dotest(A, x, b, L, C)
    A = MPI.bcast(A, 0, comm)
    b = MPI.bcast(b, 0, comm)
    x = MPI.bcast(x, 0, comm)

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
    for nblocks in (3, 5, 7, 9)
      A, x, b = buildmatrix(L, C, nblocks)
      dotest(A, x, b, L, C)
      SCM = SSCMatrix(A, L, C)
#      SCMf = factorise!(SCM; inplace=true)
      tb = @elapsed SCMf = factorise!(deepcopy(SCM); inplace=true)
      z = zeros(eltype(x), size(x))
      @test ldiv!(z, SCMf, b) ≈ x
      t1 = @elapsed ldiv!(z, SCMf, b)
      if rank == 0
        S = dropzeros!(sparse(A))
        luS = lu(S)
        ta = @elapsed lu!(luS, S)
        t0 = @elapsed (luS \ b)
        @show L, C, nblocks, t1 / t0, tb / ta
      end
    end
  end
end

MPI.Finalize()
