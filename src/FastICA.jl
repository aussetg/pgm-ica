include("preprocess.jl")

function randO(m,n)
  w = rand(m,m) - 0.5
  Q, R = qr(w)
  return w[1:m,1:n]
end

function symDecorrel(w)
  K = w*w'
  D, V = eig(K)
  return V*diagm(D.^-(0.5))*V' * w
end

function fastICA(X,c, nrounds = 50)
  # m x n = Dimensions x Samples
  # c = # of Sources
  (m,n) = size(X)
  w = randO(m,c)
  #w = ones(c,m)
  w = symDecorrel(w)
  for iters = 1:nrounds
    wx = w' * X
    #w = 1/n * X * tanh(wx)' - 1/n * (sech(wx)).^2 * ones(n) .* w
    w = 1/n * X * tanh(wx)' - w * mapslices(mean,(sech(wx)).^2,2)
    # Symetric decorelation
    w = symDecorrel(w)
  end
  return w, w' * X
end
