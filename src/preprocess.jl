function center(X)
  m = mapslices(mean,X,1)
  return X .-m , m
end

function whiten(X)
  F = eigfact(X' * X)
  D = diagm(F[:values])
  E = F[:vectors]
  return (E * D^(-0.5) * E' * X')', E, D
end
