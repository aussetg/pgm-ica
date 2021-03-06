function randO(m,n)
  c = max(m,n)
  w = rand(c,c) - 0.5
  Q, R = qr(w)
  return w[1:m,1:n]
end

@fastmath function symDecorrel(w::Matrix{Float64})
  K = w*w'
  D, V = eig(K)
  return V*diagm(D.^(-0.5))*V' * w
end

@fastmath function fastICA(X::Matrix{Float64}, c, nrounds = 50, epsilon = 1e-4)
  # m x n = Dimensions x Samples
  # c = # of Sources
  (m,n) = size(X)
  w = randO(c,m)
  w = symDecorrel(w)
  iters = 0
  error = Inf
  while iters < nrounds && error > epsilon
    wold = w
    wx = w * X
    gwx = tanh(wx)
    gwxp = ones(c,n) - gwx.^2
    w = gwx * X' - diagm(gwxp * ones(n)) * w
    # Symetric decorelation
    w = symDecorrel(w)
    error = 1 - minimum(abs(diag(w*wold')))
    iters += 1
  end
  return w, w * X
end
