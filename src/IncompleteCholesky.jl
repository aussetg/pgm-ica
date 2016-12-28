"""
    IncompleteChol(K[, eta])

Compute the incomplete cholesky decomposition of matrix K with precision eta.
If eta is not provided eta = 1e-2 is used.
Return P the permutation matrix, G a lower triangular matrix
and M the rank of the approximation such that:
|P*K*P' - G*G'| < eta
See page 20.
"""
function IncompleteChol(K::Array{Float64}, eta = 1e-2)
  N = size(K,1)
  i = 1
  Kp = K
  P = sparse(eye(N))
  G = diagm(diag(K))
  while sum(diag(G)[i:N]) > eta
    # Find new best element
    j = indmax(diag(G)[i:N]) + i - 1
    # Update P
    P[i,i] = 0
    P[j,j] = 0
    P[i,j] = 1
    P[j,i] = 1
    # Permute elements i and j in Kp
    Kp[:,i], Kp[:,j] = Kp[:,j], Kp[:,i] # En fait on peut utiliser Kp[:,[j,i]]
    Kp[i,:], Kp[j,:] = Kp[j,:], Kp[i,:]
    # Update G
    G[i,1:i], G[j,1:i] = G[j,1:i], G[i,1:i]
    G[i,i] = sqrt(diag(G)[j]) #The paper uses Kp[i,i]: doesn't work
    # ith column of G
    G[(i+1):N, i] = (Kp[(i+1):N,i] - G[(i+1):N, 1:(i-1)] * G[i, 1:(i-1)])/G[i,i]
    # Update only diagonal elements
    for k = (i+1):N
      G[k,k] = K[k,k] - sum(G[k,1:i] .^ 2)
    end
    i = i+1
  end
  return P, G[:,1:(i-1)], (i-1)::Int
end
