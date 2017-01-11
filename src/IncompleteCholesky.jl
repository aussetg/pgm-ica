"""
    IncompleteChol(K[, eta])

Compute the incomplete cholesky decomposition of matrix K with precision eta.
If eta is not provided eta = 1e-2 is used.
Return P the permutation matrix, G a lower triangular matrix
and M the rank of the approximation such that:
|P*K*P' - G*G'| < eta
See page 20.
"""
function IncompleteChol(K::Matrix{Float64}, kappa = 2*1e-3)
  # let's chose eta = 10-3 Nk / 2 as in the paper
  N = size(K,1)
  eta = 1e-3 * 0.5*N*kappa
  i = 1
  Kp = K
  P = speye(Float64,N)
  G = spdiagm(diag(K))
  while sum(diag(G)[i:N]) > eta
    # Find new best element
    j = indmax(diag(G)[i:N]) + i - 1
    # Update P
    P[:,[i,j]] = P[:,[j,i]]
    # Permute elements i and j in Kp
    Kp[:,i], Kp[:,j] = Kp[:,j], Kp[:,i] # En fait on peut utiliser Kp[:,[j,i]]
    Kp[i,:], Kp[j,:] = Kp[j,:], Kp[i,:]
    # Update G
    G[i,1:i], G[j,1:i] = G[j,1:i], G[i,1:i]
    G[i,i] = sqrt(diag(G)[j]) # The paper uses Kp[i,i]: doesn't work
    # ith column of G
    G[(i+1):N, i] = (Kp[(i+1):N,i] - G[(i+1):N, 1:(i-1)] * G[i, 1:(i-1)])/G[i,i]
    # Update only diagonal elements
    for k = (i+1):N
      G[k,k] = K[k,k] - sum(G[k,1:i] .^ 2)
    end
    i = i+1
  end
  return P, full(G[:,1:(i-1)]), (i-1)::Int
end

"""
    IncompleteChol(K,[, eta])

Compute the incomplete cholesky decomposition of the gram matrix of the
kernel K with precision eta. If eta is not provided eta = 1e-2 is used.
Elements of the gram matrix are lazily evaluated when needed, the whole Gram
matrix is never entirely constructed.
Return P the permutation matrix, G a lower triangular matrix
and M the rank of the approximation such that:
|P*K*P' - G*G'| < eta
See page 20.
"""
function IncompleteChol(K::Function, x::Vector{Float64}, kappa = 2*1e-3)
  # let's chose eta = 10-3 Nk / 2 as in the paper
  N = length(x)
  eta = 1e-3 * 0.5*N*kappa
  i = 1
  Kp = spzeros(N,N)
  P = speye(Float64,N)
  G = speye(Float64,N)
  for k = 1:N
    G[k,k] = K(x[k],x[k])
  end
  while sum(diag(G)[i:N]) > eta
    # Find new best element
    j = indmax(diag(G)[i:N]) + i - 1
    # Update P
    P[:,[i,j]] = P[:,[j,i]]
    # Construit les éléments dont on va avoir besoin
    # TODO: Faut t'il verifier si il existent deja avant de les recalculer et
    # TODO: potentiellement les ecraser ?
    for k = 1:N
      Kp[k,i] = (Kp[k,i] == 0 ? K(x[k],x[i]) : Kp[k,i])
      Kp[i,k] = Kp[k,i]
      Kp[k,j] = (Kp[k,j] == 0 ? K(x[k],x[j]) : Kp[k,j])
      Kp[j,k] = Kp[k,j]
    end
    # Permute elements i and j in Kp
    Kp[:,i], Kp[:,j] = Kp[:,j], Kp[:,i] # En fait on peut utiliser Kp[:,[j,i]]
    Kp[i,:], Kp[j,:] = Kp[j,:], Kp[i,:]
    # Update G
    G[i,1:i], G[j,1:i] = G[j,1:i], G[i,1:i]
    G[i,i] = sqrt(diag(G)[j]) # The paper uses Kp[i,i]: doesn't work
    # ith column of G
    G[(i+1):N, i] = (Kp[(i+1):N, i] - G[(i+1):N, 1:(i-1)] * G[i, 1:(i-1)])/G[i,i]
    # Update only diagonal elements
    for k = (i+1):N
      G[k,k] = K(x[k],x[k]) - sum(G[k,1:i] .^ 2)
    end
    i = i+1
  end
  return P, full(G[:,1:(i-1)]), (i-1)::Int
end
# TODO: Problème avec P, on ne remplit pas P*K*P' - G*G' < eps ?!
# TODO: On a  K - G*G' < eps
