using StructuredStaticCondensation
using Random, Test, LinearAlgebra, SparseArrays#, StatProfilerHTML
Random.seed!(0)
include("matrixbuilder.jl")
@testset "SSCCondensation" begin
  for (L, C) in ((2, 1), (3, 2), (4, 2), (16, 4), (5, 7), (128, 64), (256, 128), (1024, 512))
    for nblocks in (3, 5, 7, 9)
      A, x, b = buildmatrix(L, C, nblocks)
      SCM = SSCMatrix(A, L, C)
      tb = @elapsed SCMf = factorise!(SCM; inplace=true)
      z = zeros(eltype(x), size(x))
      @test ldiv!(z, SCMf, b) ≈ x
      t1 = @elapsed ldiv!(z, SCMf, b)
#      @profilehtml ldiv!(z, SCMf, b)
      S = dropzeros!(sparse(A))
      luS = lu(S)
      ta = @elapsed lu!(luS, S)
      luS \ b
      t0 = @elapsed luS \ b
      @show L, C, nblocks, t1 / t0, tb / ta
    end
  end
end
