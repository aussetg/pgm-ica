function centering(X)
  m = mapslices(mean,X,1)
  return X .- m , m
end

function centeringt(X)
  m = mapslices(mean,X,2)
  return X .- m , m
end

function whiten(X)
  (n,m) = size(X)
  C = 1/n * (X' * X)
  U, S, V = svd(C)
  D = S #+ 1e-7 # For statbility
  Ds = sparse(diagm(D.^(-0.5)))
  return (U * Ds * U' * X')', U, diagm(D)
end

function whiten!(X)
  (n,m) = size(X)
  C = 1/n * (X' * X)
  U, S, V = svd(C)
  D = S #+ 1e-7 # For statbility
  Ds = sparse(diagm(D.^(-0.5)))
  return (U * Ds * U' * X')'
end

function whitent(X)
  (m,n) = size(X)
  C = 1/n * (X * X')
  U, S, V = svd(C)
  D = S #+ 1e-7 # For statbility
  Ds = sparse(diagm(D.^(-0.5)))
  return (U * Ds * U' * X), U, diagm(D)
end

function whitent!(X)
  (m,n) = size(X)
  C = 1/n * (X * X')
  U, S, V = svd(C)
  D = S #+ 1e-7 # For statbility
  Ds = sparse(diagm(D.^(-0.5)))
  return (U * Ds * U' * X)
end
