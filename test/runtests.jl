using StructuredStaticCondensation
using Random, Test, LinearAlgebra
Random.seed!(0)
@testset "SSCCondensation" begin
for (L, C) in ((2, 1), (3, 2), (4, 2), (16, 4), (5, 7), (128, 64), (256, 128), (1024, 512))
  A11, A33, A55, A77 = (rand(L, L) for _ in 1:4)
  t12, t32, t34, t54, t56, t76 = (rand(L, C) for _ in 1:6)
  s21, s23, s43, s45, s65, s67 = (rand(C, L) for _ in 1:6)
  a22, a44, a66, a24, a42, a46, a64 = (rand(C, C) for _ in 1:7)
  zLL = zeros(L, L)
  zCL = zeros(C, L)
  zLC = zeros(L, C)
  zCC = zeros(C, C)
  
  A = [A11 t12 zLL;
       s21 a22 s23;
       zLL t32 A33]
  b = rand(size(A, 1))
  x = A \ b
  SCM = SSCMatrix(A, L, C)
  @test ldiv!(SCM, b; inplace=true) ≈ x
  
  A = [A11 t12 zLL zLC zLL;
       s21 a22 s23 a24 zCL;
       zLL t32 A33 t34 zLL;
       zCL a42 s43 a44 s45;
       zLL zLC zLL t54 A55]
  b = rand(size(A, 1))
  x = A \ b
  SCM = SSCMatrix(A, L, C)
  @test ldiv!(SCM, b; inplace=true) ≈ x
  
  A = [A11 t12 zLL zLC zLL zLC zLL;
       s21 a22 s23 a24 zCL zCC zCL;
       zLL t32 A33 t34 zLL zLC zLL;
       zCL a42 s43 a44 s45 a46 zCL;
       zLL zLC zLL t54 A55 t56 zLL;
       zCL zCC zCL a64 s65 a66 s67;
       zLL zLC zLL zLC zLL t76 A77]
  
  b = rand(size(A, 1))
  x = A \ b
  SCM = SSCMatrix(A, L, C)
  SCMf = factorise!(SCM)
  @test ldiv!(SCMf, b; inplace=false) ≈ x
end
end
