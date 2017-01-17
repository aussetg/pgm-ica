include("IncompleteCholesky.jl")
include("Kernels.jl")
include("preprocess.jl")
include("FastICA.jl")

# This code is inspired by the Matlab code of Stefanie Jegelka, Hao Shen, Arthur Gretton in its structure. A lot has been cleaned up and refactored to be easier to understand. Everything is native too, no C code.

"""
  gramCentering(G)

Center in the feature space a gram matrix K given as K = G*G'.
"""
function gramCentering(G::Matrix{Float64})
  N = size(G,1)
  G -= repmat(mean(G,1),N,1)
end

"""
  kernels(x, y, gamma)

Compute the kernel submatrix for x and y, vectors of differing sizes.
"""
function kernels(x::Vector{Float64}, y::Vector{Float64}, gamma::Float64)
  n = size(x)
  k = size(y)
  K = zeros(Float64, n, k)
  for i = 1:n
    for j = 1:k
      K[i, j] = gaussian(x[i], y[j], gamma)
    end
  end
  return K
end

"""
  hsic(Gs,m,N)

HSIC of the estimate represented by Gs.
"""
function hsic(Gs, m, N)
  score = 0
  for k = 1:m-1
    for l = k+1:m
      score += trace( (Gs[l]'*Gs[k])*(Gs[k]'*Gs[l]) )
    end
  end
  return score*1/N^2
end

"""
  subGrad(Knm, inds, w, x, gamma)

Compute derivative of the kernel submatrix.
"""
function subGrad(Knm, inds, w, x, gamma)
  (m, N) = size(x)
  d = size(Knm, 2)
  nd = n*d
  gamma = -gamma^2

end

"""
   hsicGrad(w, x, inds, sigma)

"""
function hsicGrad(w, x, Ms, Ps, sigma)
  (m,N) = size(x) # components x samples

  ridge = 1e-6

  y = w'*x

  # parts of Cholesky gradient
  kern_nd = Vector{Matrix{Float64}}(m)
  kern_dd = Vector{Matrix{Float64}}(m)
  dkern_nd = Vector{Matrix{Float64}}(m)

  for i=1:m
    inds = Ps[i][1:Ms[i]]
    kern_nd[i] = kernels(y[i,:], y[i,inds], gamma)
    kern_dd[i] = (kern_nd[i][inds,:] + ridge * eye(length(inds)))^(-1)
    dkern_nd[i] = subGrad(kern_nd[i], inds, w[i,:], x, gamma)
  end
  gradJ = zeros(m,m)

  # sum up pairwise gradients
  for i=1:(m-1)
    for j=(i+1):m
      g = cholGrad(kern_nd[i], kern_dd[i],dkern_nd[i], kern_nd[j], kern_dd[j], dkern_nd[j], inds[i], inds[j])
      gradJ[[i,j],:] +=  g
    end
  end
  return gradJ
end

"""
  fastkica(x::Matrix{Float64}; maxiter = 15, ε = 1e-5, gamma = 2)

Fast-KernelICA based on HSIC. See:
Shen, H., Jegelka, S., & Gretton, A. (2009). Fast Kernel-Based Independent Component Analysis. IEEE Transactions on Signal Processing, 57.
"""

function fastkica(x::Matrix{Float64}; maxiter = 15, ε = 1e-5, gamma = 2)
  m, N = size(x)
  # For our intitial guess we use FastICA
  w, _ = fastICA(x,m)
  w = w'

  w_old = w  # stores w at each iteration
  hsics = zeros(Float64, maxiter)

  for iter = 1:maxiter
      # Cholesky
      Ms = zeros(Int,m)
      Gs = Array{Array}(m) # The G matrices
      for i = 1:m
        P, G, M = IncompleteChol(gaussian, X[i,:])
        gramCentering(G)
        Ms[i] = M
        Gs[i] = G
      end

      # HSIC of current estimate
      hsics(iter) = hsic(Gs, m, N)

      # Euclidean gradient: computed entry-wise
      G = hsicGrad(w, x, inds, gamma)
      G = G/(N^2)

      # Compute approximate Hessian
      H = hsicHess(w, x, Gs, gamma);

      EG = w * G
      RG = (EG - EG') / 2 # Gradient in parameter space
      w_old = w
      w = w * expm( RG .* H ) # map Newton direction onto SO(m)

      # check for convergence
      if iter>1 && (abs(hsics(iter)-hsics(iter-1)) < ε)
          break
      end
  end
  return w, w'*x
end
