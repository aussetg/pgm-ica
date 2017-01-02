function centering(X)
  m = mapslices(mean,X,1)
  return X .- m , m
end

function centeringt(X)
  m = mapslices(mean,X,2)
  return X .- m , m
end

function whiten(X)
  C = (X' * X)
  U, S, V = svd(C)
  D = S + 1e-7
  Ds = sparse(diagm(D.^(-0.5)))
  return (U * Ds * U' * X')', U, diagm(D)
end

function whitent(X)
  C = (X * X')
  U, S, V = svd(C)
  D = S + 1e-7
  Ds = sparse(diagm(D.^(-0.5)))
  return (U * Ds * U' * X)', U, diagm(D)
end
