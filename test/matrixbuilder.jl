
function buildmatrix(L, C, nrows)
  A11, A33, A55, A77, A99 = (rand(L, L) for _ in 1:5)
  t12, t32, t34, t54, t56, t76, t79, t98 = (rand(L, C) for _ in 1:8)
  s21, s23, s43, s45, s65, s67, s87, s89 = (rand(C, L) for _ in 1:8)
  a22, a44, a66, a88, a24, a42, a46, a64, a68, a86 = (rand(C, C) for _ in 1:10)
  zLL = zeros(L, L)
  zCL = zeros(C, L)
  zLC = zeros(L, C)
  zCC = zeros(C, C)

  if nrows == 3
    A = [A11 t12 zLL;
         s21 a22 s23;
         zLL t32 A33]
    b = rand(size(A, 1))
    x = A \ b
    return A, x, b
  elseif nrows == 5
    A = [A11 t12 zLL zLC zLL;
         s21 a22 s23 a24 zCL;
         zLL t32 A33 t34 zLL;
         zCL a42 s43 a44 s45;
         zLL zLC zLL t54 A55]
    b = rand(size(A, 1))
    x = A \ b
    return A, x, b
  elseif nrows == 7
    A = [A11 t12 zLL zLC zLL zLC zLL;
         s21 a22 s23 a24 zCL zCC zCL;
         zLL t32 A33 t34 zLL zLC zLL;
         zCL a42 s43 a44 s45 a46 zCL;
         zLL zLC zLL t54 A55 t56 zLL;
         zCL zCC zCL a64 s65 a66 s67;
         zLL zLC zLL zLC zLL t76 A77]
    
    b = rand(size(A, 1))
    x = A \ b
    return A, x, b
  elseif nrows == 9
    A = [A11 t12 zLL zLC zLL zLC zLL zLC zLL;
         s21 a22 s23 a24 zCL zCC zCL zCC zCL;
         zLL t32 A33 t34 zLL zLC zLL zLC zLL;
         zCL a42 s43 a44 s45 a46 zCL zCC zCL;
         zLL zLC zLL t54 A55 t56 zLL zLC zLL;
         zCL zCC zCL a64 s65 a66 s67 a68 zCL;
         zLL zLC zLL zLC zLL t76 A77 t79 zLL;
         zCL zCC zCL zCC zCL a86 s87 a88 s89;
         zLL zLC zLL zLC zLL zLC zLL t98 A99;]
    
    b = rand(size(A, 1))
    x = A \ b
    return A, x, b
  else
    error("invalid nrows")
  end
end
