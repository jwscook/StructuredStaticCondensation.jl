using MPI
MPI.Init()

include("mpiinclude.jl")
include("matrixbuilder.jl")

using StructuredStaticCondensation

function makesharedmatrix(sharedrank, sharedcomm, tmp)
  dimslocal = sharedrank == 0 ? size(tmp) : (0, 0)
  win, arrayptr = MPI.Win_allocate_shared(Array{Float64}, prod(dimslocal), sharedcomm)
  MPI.Barrier(sharedcomm)
  A = MPI.Win_shared_query(Array{Float64}, prod(size(tmp)), win; rank=0)
  A = reshape(A, size(tmp))
  sharedrank == 0 && (A .= tmp)
  MPI.Barrier(sharedcomm)
  return A, win
end

using Random, Test, LinearAlgebra

@testset "SSCCondensation" begin
  comm = MPI.COMM_WORLD
  commsize = MPI.Comm_size(comm)
  rank = MPI.Comm_rank(comm)
  Random.seed!(rank)
  color = rank < commsize ÷ 2 ? 0 : 1 # split into two shared memory mpi communicators
  sharedcomm = MPI.Comm_split(comm, color, rank)
  sharedrank = MPI.Comm_rank(sharedcomm)
  sharedsize = MPI.Comm_size(sharedcomm)

  function dotest(Atmp, x, b, L, C)
    A, win = makesharedmatrix(sharedrank, sharedcomm, Atmp)
    b = MPI.bcast(b, 0, sharedcomm)
    x = MPI.bcast(x, 0, sharedcomm)
    SCM = SSCMatrix(A, L, C)
    StructuredStaticCondensation.distributeenumerations!(SCM, sharedrank, sharedsize)
    sleep(rank)
    @test ldiv!(SCM, b; callback=MPICallBack(sharedcomm, sharedrank, sharedsize)) ≈ x
    MPI.free(win)
  end

  for (L, C) in ((3, 2), (4, 2), (16, 4), (5, 7), (128, 64), (256, 128), (1024, 512))
    for nblocks in (3, 5, 7, 9)
      A, x, b = buildmatrix(L, C, nblocks)
      dotest(A, x, b, L, C)
      SCM = SSCMatrix(A, L, C)
      SCMf = factorise!(SCM; inplace=true)
      z = zeros(eltype(x), size(x))
      @test ldiv!(z, SCMf, b; inplace=true) ≈ x
    end
  end
end

MPI.Finalize()
