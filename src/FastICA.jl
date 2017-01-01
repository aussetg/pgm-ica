include("preprocess.jl")

function fastICA(X,c)
  # n x m = Samples x Dimensions
  # c = Sources
  (n,m) = size(X)
  w = rand((m,c))
  #w = ones(m,c)
  for p = 1:c
    for iters = 1:Int64(50)
      w[p,:] = 1/n * X' * tanh(w[p,:]' * X')' - 1/n * ((sech(w[p,:]' * X')).^2 * ones(n) * w[p,:]')'
      sw = zeros(m)
      for j = 1:(p-1)
        sw += (w[p,:]' * w[j,:]) .* w[j,:]
      end
      w[p,:] = w[p,:] - sw
      w[p,:] = w[p,:]/norm(w[p,:])
    end
  end
  return w, (w * X')'
end
