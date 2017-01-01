include("IncompleteCholesky.jl")
include("preprocess.jl")
include("FastICA.jl")

"""
Helper function to create the Gram matrix associated to a Gaussian symmetrix
kernel and parameter sigma.
"""
function gramGauss(x, gamma = 2)
  N = size(x,1)
  K = eye(N)
  for i = 1:N
    for j = 1:(i-1)
      K[i,j] = exp(-gamma*(x[i]-x[j])^2)
      K[j,i] = K[i,j]
    end
  end
  # TODO: PAGE 11 SAYS WE NEED TO CENTER THE GRAM MATRIX !
  # TODO: DON'T FORGET TO DO IT
  N0 = eye(N) - 1/N * ones(N,N)
  return N0*K*N0
end

"""
NOT YET IMPLEMENTED
KCCA contrast function
"""
function kcca()
end

"""
Helper function to apply the regularization to the the matrix
"""
function regularize(x,N::Int, k = 1e-3)
  return x/(x + 0.5*N*k)
end

"""
  updateJ(s,i,j)

This function comute a new contrast (only for KGV at the moment)
using the fact that the Gram matrix only changes column wise for each
partial derivative.
"""
function updateJ(s,i,j,Rs,Us,Ms)
  if (i>j)
    i, j = j, i
  end
  Msp = Ms # New sizes
  Rsp = Rs
  Usp = Us
  m, N = size(s)
  # First we will compute the updated values for i and j
  Ki = gramGauss(s[i,:])
  Kj = gramGauss(s[j,:])
  Pi, Gi, Mi = IncompleteChol(Ki)
  Msp[i] = Mi
  Pj, Gj, Mj = IncompleteChol(Ki)
  Msp[j] = Mj
  Ui, Si, Vi = svd(Gi)
  Usp[i] = Ui
  Li = sparse(diagm(Si.^2))
  Ri = regularize.(Li,N)
  Rsp[i] = Ri
  Uj, Sj, Vj = svd(Gj)
  Usp[j] = Uj
  Lj = sparse(diagm(Sj.^2))
  Rj = regularize.(Lj,N)
  Rsp[j] = Rj
  # Now we will build Rk
  Rk = eye(sum(Msp))
  ix = cumsum(Msp) - Msp[1] + 1
  for i = 2:m
    for j = 1:(i-1)
      mat = Rsp[i] * Usp[i]' * Usp[j] * Rsp[j]
      Rk[ix[i]:(ix[i]+Msp[i]-1),ix[j]:(ix[j]+Msp[j]-1)] = mat
      Rk[ix[j]:(ix[j]+Msp[j]-1),ix[i]:(ix[i]+Msp[i]-1)] = mat'
    end
  end
  return -0.5*log(det(Rk))
end

"""
  finiteD(x,w[, ε])

This function compute finite differences (only for KGV at the moment)
using the fact that the Gram matrix only changes column wise for each
partial derivative.
"""
function finiteD(x, w , ε = 1e-3)
  m, N = size(w)
  wd = zeros(m,N)
  s0 = w' * x
  J0, Rs, Us, Ms = kgvc!(s0)
  for i = 1:(m-1)
    for j = (i+1):m
      s = s0
      s[[i, j],:] = [cos(ε) sin(ε) ; sin(-ε) cos(ε)]*s[[i, j],:]
      J = updateJ(s,i,j,Rs,Us,Ms)
      wd[i,j] = (J-J0)/ε
      wd[j,i] = -(J-J0)/ε
    end
  end
  return J0, w*wd
end

"""
  kgvc(X[, k])

"""
function kgvc(X, k = 2*1e-3)
  m, N = size(X) # Components, Observations
  Ms = zeros(Int,m)
  Rs = Array{Array}(m) # The R matrices
  Us = Array{Array}(m) # The U matrices
  for i = 1:m
    K = gramGauss(X[i,:])
    # TODO: PAGE 11 SAYS WE NEED TO CENTER THE GRAM MATRIX !
    # TODO: DON'T FORGET TO DO IT
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
  ix = cumsum(Ms) - Ms + 1
  for i = 2:m
    for j = 1:(i-1)
      mat = Rs[i] * Us[i]' * Us[j] * Rs[j]
      Rk[ix[i]:(ix[i]+Ms[i]-1),ix[j]:(ix[j]+Ms[j]-1)] = mat
      Rk[ix[j]:(ix[j]+Ms[j]-1),ix[i]:(ix[i]+Ms[i]-1)] = mat'
    end
  end
  return -0.5*log(det(Rk))
end

"""
  kgvc!(X[, k])

Same as kgvc except returns additional informations needed to rebuild Rk
"""
function kgvc!(X, k = 2*1e-3)
  m, N = size(X) # Components, Observations
  Ms = zeros(Int,m)
  Rs = Array{Array}(m) # The R matrices
  Us = Array{Array}(m) # The U matrices
  for i = 1:m
    K = gramGauss(X[i,:])
    # TODO: PAGE 11 SAYS WE NEED TO CENTER THE GRAM MATRIX !
    # TODO: DON'T FORGET TO DO IT
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
  ix = cumsum(Ms) - Ms + 1
  for i = 2:m
    for j = 1:(i-1)
      mat = Rs[i] * Us[i]' * Us[j] * Rs[j]
      Rk[ix[i]:(ix[i]+Ms[i]-1),ix[j]:(ix[j]+Ms[j]-1)] = mat
      Rk[ix[j]:(ix[j]+Ms[j]-1),ix[i]:(ix[i]+Ms[i]-1)] = mat'
    end
  end
  return -0.5*log(det(Rk)), Rs, Us, Ms
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
function geod(w,h,t)
  A=w'*h
  A=0.5*(A-A')
  MN=expm(t*A)
  wt=w*MN
  return wt, h*MN
end

"""
  initializeSearch(w, d, x, a, b, Ja)

Find inital conditions for the linesearch.
"""
function initializeSearch(w, d, x, a, b, Ja)
  φ = 1.618
  ε = 1e-10
  wp, hp = geod(w,d,a)
  fa = kgvc(wp' * x)
  wp, hp = geod(w,d,b)
  fb = kgvc(wp' * x)
  if (fb > fa)
     a, b = b, a
     fa, fb = fb, fa
  end
  c = b + φ * (b-a)
  wp, hp = geod(w,d,c)
  fc = kgvc(wp' * x)

  while (fb > fc)
   r = (b-a)*(fb-fc);
   q = (b-c)*(fb-fa);
   u = b-((b-c)*q-(b-a)*r)/(2.0*max(abs(q-r),ε)*sign(q-r))
   ulim = b+100*(c-b)
   if ((b-u)*(u-c) > 0)
      wp, hp = geod(w,d,u)
      fu = kgvc(wp'*x)

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
      fu = kgvc(wp'*x)
   else
      if ((c-u)*(u-ulim) > 0)
         wp, hp = geod(w,d,u)
         fu = kgvc(wp'*x)
         if (fu < fc)
            b = c
            c = u
            u = c + 100*(c-b)

            fb = fc
            fc = fu
            wp, hp = geod(w,d,u)
            fu = kgvc(wp'*x)
         end
      else
         if ((u-ulim)*(ulim-c) >= 0)

            u = ulim
            wp, hp = geod(w,d,u)
            fu = kgvc(wp'*x)

         else
            u = c + 100*(c-b)
            wp, hp = geod(w,d,u)
            fu = kgvc(wp'*x)
         end
      end
   end
   a, b ,c = b, c, u
   fa, fb, fc = fb, fc, fu
end
return a,b,c,fa,fb,fc
end

"""
  linesearch(w,d,x,a,b,c)

Perform a linesearch by golden section.
Not exactly a linesearch per se, but equivalent on a geodesic of the
Stiefel manifold.
"""
function linesearch(w,d,x,a,b,c, tol = 1e-2)
  φ = 1.618
  C = 0.382
  R = 0.618
  resφ = 0.382
  x0 = a
  x3 = c
  maxiter = 100

  if (abs(c-b) > abs(b-a))
    x1 = b
    x2 = b + C*(c-b)
  else
    x2 = b
    x1 = b - C*(b-a)
  end
  wp, hp = geod(w,d,x1)
  f1 = kgvc(wp'*x)
  wp, hp = geod(w,d,x2)
  f2 = kgvc(wp'*x)
  k = 1
  while ((abs(x3-x0) > tol) & (k<maxiter))
    if f2 < f1
      x0 = x1
      x1 = x2
      x2 = R*x1 + C*x3
      f1 = f2
      wp, hp = geod(w,d,x2)
      f2 = kgvc(wp'*x)
    else
      x3 = x2
      x2 = x1
      x1 = R*x2 + C*x0
      f2 = f1
      wp, hp = geod(w,d,x1)
      f1 = kgvc(wp'*x)
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
  kgv(X)

Perform ICA using the KGV contrast function and using gradient descent
"""
function kgv(X)
  m, N = size(X)
  # For our intitial guess we use FastICA
  w, _ = fastICA(X',m)
  w = w'
  #w = eye(m)
  x = X
  ε = 1e-5
  err = 1
  maxiter = 15
  iters = 0
  b = 1
  while (err > ε) && iters < maxiter
    J, grad = finiteD(x,w)
    d = grad

    gap = sqrt(0.5*trace(grad'*grad))

    ## Initialize interval
    a,b,c,fa,fb,fc = initializeSearch(w,d,x,0,b,J)
    tol=max(abs(ε/gap), abs(mean([a, b, c])/10))
    ## Search on the interval
    b, Jmin = linesearch(w,d,x,a,b,c,tol)
    w, h = geod(w,d,b)
    err = abs(Jmin/J - 1)
    iters = iters + 1
    J = Jmin
  end
  return w, w'*xw
end
