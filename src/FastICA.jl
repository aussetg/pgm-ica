include("preprocess.jl")

function fastICA(X,c, tol = 1e-2)
  Xc, m = center(X)
  Xw , E, D = whiten(Xc)
  (n,m) = size(X)
  w = rand((c,n))
  for p = 1:c
    while true
      old_wp = w[p,:]
      w[p,:] = 1/m * Xw * tanh(w[p,:]' * Xw)' - 1/m * (sech(w[p,:]' * Xw)').^2 * ones(m)' * w[p,:]
      sw = zeros(n)
      for j = 1:(p-1)
        sw += (w[p,:]' * w[j,:]) .* w[j,:]
      end
      w[p,:] = w[p,:] - sw
      w[p,:] = w[p,:]/norm(w[p,:])
      if norm(old_wp - w[p,:]) <= tol
        break
      end
    end
  end
  Xuw = (E * D^2 * E' * Xw')'
  Ainv = (E * (D.^2) * E' * w')'
  return Ainv * Xw + w' * m
end

fastICA(rand(10,10),2)
