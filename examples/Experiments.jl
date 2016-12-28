function center(X)
  m = mapslices(mean,X,1)
  return X .-m , m
end

function whiten(X)
  F = eigfact(X' * X)
  D = diagm(F[:values])
  E = F[:vectors]
  return (E * D^(-0.5) * E' * X')', E, D
end

function fastICA(X,c)
  Xc, mx = center(X)
  Xw , E, D = whiten(Xc)
  # n x m = Samples x Dimensions
  # c = Sources
  (n,m) = size(X)
  w = ones(m,c)
  for p = 1:c
    for iters = 1:5
      w[p,:] = 1/n * Xw' * tanh(w[p,:]' * Xw')' - 1/n * ((sech(w[p,:]' * Xw')).^2 * ones(n) * w[p,:]')'
      sw = zeros(m)
      for j = 1:(p-1)
        sw += (w[p,:]' * w[j,:]) .* w[j,:]
      end
      w[p,:] = w[p,:] - sw
      w[p,:] = w[p,:]/norm(w[p,:])
    end
  end
  return w, (w * Xw')'
end

using Distributions
import Plots
Plots.gr()

u = Uniform(5,10)

s1, s2 = rand(u,Int64(1e3)), rand(u,Int64(1e3))

s = [s1 s2]

Plots.scatter(s[:,1],s[:,2],leg=false,border=false, markersize=1)
Plots.savefig("figures/s_orig")

A = [1 2; 21 20]

x = (A * s')'

Plots.scatter(x[:,1],x[:,2],leg=false,border=false, markersize=1)
Plots.savefig("figures/x")

xc , m = center(x)

Plots.scatter(xc[:,1],xc[:,2],leg=false,border=false, markersize=1)
Plots.savefig("figures/x_c")

xw, E , D = whiten(xc)

Plots.scatter(xw[:,1],xw[:,2],leg=false,border=false, markersize=1)
Plots.savefig("figures/x_w")

w, s_ica = fastICA(x,2)

Plots.scatter(s_ica[:,1],s_ica[:,2],leg=false,border=false, markersize=1)
Plots.savefig("figures/s_retrieved")

include("../src/KernelICA.jl")

###### Trying things
w = kgv(xw')
s = (w^-1) * xw'
Plots.scatter(xw'[1,:],xw'[2,:],leg=false,border=false, markersize=1)
Plots.scatter(s[1,:],s[2,:],leg=false,border=false, markersize=1)

#### A more interesting Distribution

T = TDist(1.5)
s1b = rand(T,Int(1e4))
s2b = rand(T,Int(1e4))
sb = [s1b s2b]'
Ab = [1 1; 0 2]
xb = Ab * sb

Plots.scatter(sb[1,:],sb[2,:],leg=false,border=false, markersize=1)
Plots.scatter(xb[1,:],xb[2,:],leg=false,border=false, markersize=1)

xbc, mb = center(xb)
xbw, Eb, Db = whiten(xbc)
