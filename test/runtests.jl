using StructuredStaticCondensation
using Random, Test, LinearAlgebra, SparseArrays, StatProfilerHTML, Profile
using BenchmarkTools
Random.seed!(0)
include("matrixbuilder.jl")
@testset "SSCCondensation" begin
  nreps = 1
  for (L, C) in ((2, 1), (3, 2), (4, 2), (16, 4), (5, 7), (128, 64), (256, 128), (1024, 512))
    for nblocks in (3, 5, 7, 9)
      A, x, b = buildmatrix(L, C, nblocks)
      SCM = SSCMatrix(A, L, C)
      SCMf = factorise!(SCM; inplace=true)
      tb = minimum((q = deepcopy(SCM); @elapsed factorise!(q; inplace=false)) for _ in 1:nreps)
      z = zeros(eltype(x), size(x))
      @test ldiv!(z, SCMf, b) ≈ x
      t1 = minimum((q=deepcopy(SCMf); @elapsed ldiv!(z, q, b)) for _ in 1:nreps)
      Profile.init(n=10^7, delay=0.00001);
      Profile.clear()
      #@profilehtml ldiv!(z, SCMf, b)

      bb = copy(b); @profilehtml ldiv!(SCMf, bb); z .= bb
      S = dropzeros!(sparse(A))
      luS = lu(S)
      ta = minimum((q = deepcopy(luS); @elapsed lu!(q, S)) for _ in 1:nreps)
      luS \ b
      t0 = minimum((q = deepcopy(luS); @elapsed q \ b) for _ in 1:nreps)
      @show L, C, nblocks, t1 / t0, tb / ta
    end
  end
end
