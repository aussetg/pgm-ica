include("preprocess.jl")

function fastICA(X,c)
  Xc, mx = center(X)
  Xw , E, D = whiten(Xc)
  # n x m = Samples x Dimensions
  # c = Sources
  (n,m) = size(X)
  w = rand((m,c))
  for p = 1:c
    for iters = 1:Int64(1e4)
      w[p,:] = 1/n * Xw' * tanh(w[p,:]' * Xw')' - 1/n * ((sech(w[p,:]' * Xw')).^2 * ones(n) * w[p,:]')'
      sw = zeros(m)
      for j = 1:(p-1)
        sw += (w[p,:]' * w[j,:]) .* w[j,:]
      end
      w[p,:] = w[p,:] - sw
      w[p,:] = w[p,:]/norm(w[p,:])
    end
  end
  return w, (w * Xw')'
end
