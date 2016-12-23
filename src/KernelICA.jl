include("IncompleteCholesky.jl")

function gramGauss(x, sigma = 1e-2)
  N = size(x,1)
  K = Matrix{Float64}(N,N)
  for i = 1:N
    for j = 1:i
      K[i,j] = exp(-sigma*(x[i]-x[j])^2)
      K[j,i] = exp(-sigma*(x[i]-x[j])^2)
    end
  end
  return K
end

function kcca()
end

function regularize(x,N::Int, k = 1e-3)
  return x/(x + 0.5*N*k)
end

function kgv(X, k = 1e-3)
  N=size(X,2) # Observations
  m=size(X,1) # Components
  Ms = zeros(Int,m)
  Rs = Array{Array}(m) # The R matrices
  Us = Array{Array}(m) # The U matrices
  for i = 1:m
    gram = gramGauss(X[i,:])
    P, G, M = IncompleteChol(K)
    U,S,V = svd(G)
    L = sparse(diagm(S.^2))
    R = regularize.(L,N,k)
    Rs[i] = R
    Us[i] = U
    Ms[i] = M
  end
  # Build Rk
  Rk = eye(sum(Ms))
  ix = cumsum(Ms) - Ms[1] + 1
  for i = 2:m
    for j = 1:(i-1)
      mat = Rs[i] * Us[i]' * Us[j] * Rs[j]
      Rk[ix[i]:(ix[i]+Ms[i]-1),ix[j]:(ix[j]+Ms[j]-1)] = mat
      Rk[ix[j]:(ix[j]+Ms[j]-1),ix[i]:(ix[i]+Ms[i]-1)] = mat'
    end
  end
  return -0.5*log(det(Rk))
end

kgv(X)
