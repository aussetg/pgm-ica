include("preprocess.jl")

function randO(m,n)
  w = rand(m,m) - 0.5
  Q, R = qr(w)
  return w[1:m,1:n]
end

function fastICA(X,c, nrounds = 50)
  # n x m = Samples x Dimensions
  # c = Sources
  (n,m) = size(X)
  Xp = X'
  w = randO(m,c)
  #w = ones(m,c)
  for p = 1:c
    for iters = 1:nrounds
      w[:,p] = 1/n * Xp * tanh(w[p,:] * Xp)' - 1/n * ((sech(w[p,:] * Xp)).^2 * ones(n) * w[p,:])'
      sw = zeros(m)
      for j = 1:(p-1)
        sw += (w[p,:] * w[:,j]) .* w[:,j]
      end
      w[:,p] = w[:,p] - sw
      w[:,p] = w[:,p]/norm(w[:,p])
    end
  end
  return w, (w' * Xp)'
end
