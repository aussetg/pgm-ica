### See:
### Bach, F. R., & Jordan, M. I. (2002).
### Kernel Independent Component Analysis.
### Journal of Machine Learning Research, 3, 1–48.
### https://doi.org/10.1162/153244303768966085


"""
  gramCentering(G)

Center in the feature space a gram matrix K given as K = G*G'.
"""
function gramCentering(G::Matrix{Float64})
  N = size(G,1)
  G -= repmat(mean(G,1),N,1)
end

"""
  gram(K::Function, x::Vector{Float64})

Helper function to create the Gram matrix associated to a kernel K.
"""
function gram(K::Function, x::Vector{Float64})
  N = length(x)
  Km = eye(N)
  for i = 1:N
    for j = 1:(i-1)
      Km[i,j] = K(x[i],x[j])
      Km[j,i] = Km[i,j]
    end
  end
  # TODO: PAGE 11 SAYS WE NEED TO CENTER THE GRAM MATRIX !
  # TODO: DON'T FORGET TO DO IT
  N0 = eye(N) - 1/N * ones(N,N)
  # For some mysterious reasons, N0*K*N0 isn't symetric for Julia.
  # Probably a numerical problem. So we fix that
  K = N0*Km*N0
  K = 0.5*(Km+Km')
  return K
end

"""
  regularize(x, N::Int[, k = 1e-3])

Helper function to apply the regularization to the the matrix
"""
function regularize(x, N::Int, k = 1e-3)
  return x/(x + 0.5*N*k)
end

"""
  updateJ(s, i, j, Rs, Us, Ms, C, K)

This function comute a new contrast (only for KGV at the moment)
using the fact that the Gram matrix only changes column wise for each
partial derivative.
"""
function updateJ(s::Matrix{Float64}, i, j, Rs, Us, Ms, C::Function, K::Function)
  if (i>j)
    i, j = j, i
  end
  Msp = Ms # New sizes
  Rsp = Rs
  Usp = Us
  m, N = size(s)
  # First we will compute the updated values for i and j
  for k = [i, j]
    Pk, Gk, Mk = IncompleteChol(K, s[k,:])
    gramCentering(Gk)
    Msp[k] = Mk
    Uk, Sk, Vk = svd(Gk)
    Usp[k] = Uk
    Lk = sparse(diagm(Sk.^2))
    Rk = regularize.(Lk,N)
    Rsp[k] = Rk
  end
  # Now we will build Rk
  Rk = eye(sum(Msp))
  ix = cumsum(Msp) - Msp + 1
  for i = 2:m
    for j = 1:(i-1)
      mat = Rsp[i] * Usp[i]' * Usp[j] * Rsp[j]
      Rk[ix[i]:(ix[i]+Msp[i]-1),ix[j]:(ix[j]+Msp[j]-1)] = mat
      Rk[ix[j]:(ix[j]+Msp[j]-1),ix[i]:(ix[i]+Msp[i]-1)] = mat'
    end
  end
  return C(Rk)
end

"""
  finiteD(x,w, C, K[, ε])

This function compute finite differences (only for KGV at the moment)
using the fact that the Gram matrix only changes column wise for each
partial derivative.
"""
function finiteD(x::Matrix{Float64}, w::Matrix{Float64}, C::Function, K::Function, ε = 1e-3)
  m, N = size(w)
  wd = zeros(m,N)
  s0 = w' * x
  J0, Rs, Us, Ms = contrast!(s0, C, K)
  for i = 1:(m-1)
    for j = (i+1):m
      s = s0
      s[[i, j],:] = [cos(ε) sin(ε) ; sin(-ε) cos(ε)]*s[[i, j],:]
      J = updateJ(s, i, j, Rs, Us, Ms, C, K)
      wd[i,j] = (J-J0)/ε
      wd[j,i] = -(J-J0)/ε
    end
  end
  return J0, w*wd
end

"""
  contrast(X, C, K[, k])

"""
function contrast(X::Matrix{Float64}, C::Function, K::Function, k = 2*1e-3)
  m, N = size(X) # Components, Observations
  Ms = zeros(Int,m)
  Rs = Array{Array}(m) # The R matrices
  Us = Array{Array}(m) # The U matrices
  for i = 1:m
    # TODO: PAGE 11 SAYS WE NEED TO CENTER THE GRAM MATRIX !
    # TODO: DON'T FORGET TO DO IT
    P, G, M = IncompleteChol(K, X[i,:])
    gramCentering(G)
    U,S,V = svd(G)
    L = sparse(diagm(S.^2))
    R = regularize.(L,N,k)
    Rs[i] = R
    Us[i] = U
    Ms[i] = M
  end
  # Build Rk
  Rk = eye(sum(Ms))
  ix = cumsum(Ms) - Ms + 1
  for i = 2:m
    for j = 1:(i-1)
      mat = Rs[i] * Us[i]' * Us[j] * Rs[j]
      Rk[ix[i]:(ix[i]+Ms[i]-1),ix[j]:(ix[j]+Ms[j]-1)] = mat
      Rk[ix[j]:(ix[j]+Ms[j]-1),ix[i]:(ix[i]+Ms[i]-1)] = mat'
    end
  end
  return C(Rk)
end

"""
  contrast!(X, C, K[, k])

Same as kgvc except returns additional informations needed to rebuild Rk
"""
function contrast!(X::Matrix{Float64}, C::Function, K::Function, k = 2*1e-3)
  m, N = size(X) # Components, Observations
  Ms = zeros(Int,m)
  Rs = Array{Array}(m) # The R matrices
  Us = Array{Array}(m) # The U matrices
  for i = 1:m
    # TODO: PAGE 11 SAYS WE NEED TO CENTER THE GRAM MATRIX !
    # TODO: DON'T FORGET TO DO IT
    P, G, M = IncompleteChol(K, X[i,:])
    gramCentering(G)
    U,S,V = svd(G)
    L = sparse(diagm(S.^2))
    R = regularize.(L,N,k)
    Rs[i] = R
    Us[i] = U
    Ms[i] = M
  end
  # Build Rk
  Rk = eye(sum(Ms))
  ix = cumsum(Ms) - Ms + 1
  for i = 2:m
    for j = 1:(i-1)
      mat = Rs[i] * Us[i]' * Us[j] * Rs[j]
      Rk[ix[i]:(ix[i]+Ms[i]-1),ix[j]:(ix[j]+Ms[j]-1)] = mat
      Rk[ix[j]:(ix[j]+Ms[j]-1),ix[i]:(ix[i]+Ms[i]-1)] = mat'
    end
  end
  return C(Rk), Rs, Us, Ms
end

"""
  geod(w,h,t)

Geodesic on the Stiefel Manifold (i.e W such as W' W = I).
w the matrix, h the tangent to the manifold (skew symetric matrices).
See: Edelman, A., Arias, T. A., & Smith, S. T. (1998).
The Geometry of Algorithms with Orthogonality Constraints.
SIAM Journal on Matrix Analysis and Applications, 20(2), 303–353.
https://doi.org/10.1137/S0895479895290954
"""
function geod(w::Matrix{Float64}, h::Matrix{Float64}, t)
  A=w'*h
  A=0.5*(A-A')
  MN=expm(t*A)
  wt=w*MN
  return wt, h*MN
end

"""
  initializeSearch(w, d, x, a, b, Ja, C, K)

Find inital conditions for the linesearch.
Mostly follows the scheme used by: Bach, F. R., & Jordan, M. I. (2002). Kernel Independent Component Analysis. Journal of Machine Learning Research, 3, 1–48. https://doi.org/10.1162/153244303768966085
"""
function initializeSearch(w::Matrix{Float64}, d, x, a, b, Ja, C::Function, K::Function)
  φ = 1.618
  ε = 1e-10
  wp, hp = geod(w,d,a)
  fa = contrast(wp' * x, C, K)
  wp, hp = geod(w,d,b)
  fb = contrast(wp' * x, C, K)
  if (fb > fa)
     a, b = b, a
     fa, fb = fb, fa
  end
  c = b + φ * (b-a)
  wp, hp = geod(w,d,c)
  fc = contrast(wp' * x, C, K)

  while (fb > fc)
   r = (b-a)*(fb-fc);
   q = (b-c)*(fb-fa);
   u = b-((b-c)*q-(b-a)*r)/(2.0*max(abs(q-r),ε)*sign(q-r))
   ulim = b+100*(c-b)
   if ((b-u)*(u-c) > 0)
      wp, hp = geod(w,d,u)
      fu = contrast(wp'*x, C, K)

      if (fu < fc)
         a = b
         b = u
         fa = fb
         fb = fu
         break
      else
         if (fu > fb)
            c = u
            fc = fu
            break
         end
      end
      u = c + 100*(c-b)
      wp, hp = geod(w,d,u)
      fu = contrast(wp'*x, C, K)
   else
      if ((c-u)*(u-ulim) > 0)
         wp, hp = geod(w,d,u)
         fu = contrast(wp'*x, C, K)
         if (fu < fc)
            b = c
            c = u
            u = c + 100*(c-b)

            fb = fc
            fc = fu
            wp, hp = geod(w,d,u)
            fu = contrast(wp'*x, C, K)
         end
      else
         if ((u-ulim)*(ulim-c) >= 0)

            u = ulim
            wp, hp = geod(w,d,u)
            fu = contrast(wp'*x, C, K)

         else
            u = c + 100*(c-b)
            wp, hp = geod(w,d,u)
            fu = contrast(wp'*x, C, K)
         end
      end
   end
   a, b ,c = b, c, u
   fa, fb, fc = fb, fc, fu
end
return a,b,c,fa,fb,fc
end

"""
  linesearch(w, d, x, a, b, c, C, K[, tol = 1e-2])

Perform a linesearch by golden section.
Not exactly a linesearch per se, but equivalent on a geodesic of the
Stiefel manifold.

Mostly follows the scheme used by: Bach, F. R., & Jordan, M. I. (2002). Kernel Independent Component Analysis. Journal of Machine Learning Research, 3, 1–48. https://doi.org/10.1162/153244303768966085
"""
function linesearch(w::Matrix{Float64}, d, x, a, b, c, C::Function, K::Function, tol = 1e-2)
  φ = 1.618
  p = 0.382
  R = 0.618
  resφ = 0.382
  x0 = a
  x3 = c
  maxiter = 100

  if (abs(c-b) > abs(b-a))
    x1 = b
    x2 = b + p*(c-b)
  else
    x2 = b
    x1 = b - p*(b-a)
  end
  wp, hp = geod(w, d, x1)
  f1 = contrast(wp'*x, C, K)
  wp, hp = geod(w, d, x2)
  f2 = contrast(wp'*x, C, K)
  k = 1
  while ((abs(x3-x0) > tol) & (k<maxiter))
    if f2 < f1
      x0 = x1
      x1 = x2
      x2 = R*x1 + p*x3
      f1 = f2
      wp, hp = geod(w,d,x2)
      f2 = contrast(wp'*x, C, K)
    else
      x3 = x2
      x2 = x1
      x1 = R*x2 + p*x0
      f2 = f1
      wp, hp = geod(w, d, x1)
      f1 = contrast(wp'*x, C, K)
    end
    k = k+1;
  end
  if f1 < f2
    xmin = x1
    fmin = f1
  else
    xmin = x2
    fmin = f2
  end
  return xmin, fmin
end

"""
  kica(X,[C = kgv, K = gaussian])

Perform ICA using the contrast function C and using steepest descent.
The contrast is computed with respect to the kernel K of signature:
K :: Float -> Float -> Float
"""
function kica(x::Matrix{Float64}; C::Function = kgv, K::Function = gaussian, maxiter = 15, ε = 1e-5)
  m, N = size(x)
  # For our intitial guess we use FastICA
  w, _ = fastICA(x,m)
  w = w'
  err = 1
  iters = 0
  b = 1
  while (err > ε) && iters < maxiter
    J, grad = finiteD(x, w, C, K)
    d = grad

    gap = sqrt(0.5*trace(grad'*grad))

    ## Initialize interval
    a,b,c,fa,fb,fc = initializeSearch(w, d, x, 0, b, J, C, K)
    tol=max(abs(ε/gap), abs(mean([a, b, c])/10))
    ## Search on the interval
    b, Jmin = linesearch(w, d, x, a, b, c, C, K, tol)
    w, h = geod(w,d,b)
    err = abs(Jmin/J - 1)
    iters = iters + 1
    J = Jmin
  end
  return w', w'*x
end

"""
  kgv(Rk)

Compute the KGV contrast function from the Rk matrix
"""
function kgv(Rk)
  return -0.5*logdet(Rk)
end

"""
  kcca(Rk)

Compute the KCCA contrast function from the Rk matrix
"""
function kcca(Rk)
  return -0.5*log(eigmin(Rk))
end
