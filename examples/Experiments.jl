function centering(X)
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
  Xc, mx = centering(X)
  Xw , E, D = whiten(Xc)
  # n x m = Samples x Dimensions
  # c = Sources
  (n,m) = size(X)
  w = ones(m,c)
  for p = 1:c
    for iters = 1:10
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

xc , m = centering(x)

Plots.scatter(xc[:,1],xc[:,2],leg=false,border=false, markersize=1)
Plots.savefig("figures/x_c")

xw, E , D = whiten(xc)

Plots.scatter(xw[:,1],xw[:,2],leg=false,border=false, markersize=1)
Plots.savefig("figures/x_w")

w, s_ica = fastICA(x,2)

Plots.scatter((w'*xw')[1,:],(w'*xw')[2,:],leg=false,border=false, markersize=1)
Plots.savefig("figures/s_retrieved")

include("../src/KernelICA.jl")

###### Trying things
w = kgv(xw')
s = (w^-1) * xw'
Plots.scatter(xw'[1,:],xw'[2,:],leg=false,border=false, markersize=1)
Plots.scatter(s[1,:],s[2,:],leg=false,border=false, markersize=1)

#### A more interesting Distribution

using Distributions
using Plots
gr()

T = Truncated(TDist(1.5),-20,20)
s1b = rand(T,Int(1e3))
s2b = rand(T,Int(1e3))
sb = [s1b s2b]'
Ab = [1 1; 0 2]
xb = Ab * sb

Plots.scatter(sb[1,:],sb[2,:],leg=false,border=false, markersize=1)
Plots.scatter(xb[1,:],xb[2,:],leg=false,border=false, markersize=1)

xbc, mb = centering(xb')
xbw, Eb, Db = whiten(xbc)
xbw = xbw'

w, sr = kgv(xb)

w = eye(2)

finiteD(xbw,w)

w, sr = kgv(xbw)
wf, sf = fastICA(xbw',2)
Plots.scatter(sf[:,1],sf[:,2],leg=false,border=false, markersize=1)
Plots.savefig("figures/sb_ica_orig")

Plots.scatter(sb[1,:],sb[2,:],leg=false,border=false, markersize=1)
Plots.savefig("figures/sb_orig")
Plots.scatter(xb[1,:],xb[2,:],leg=false,border=false, markersize=1)
Plots.savefig("figures/xb")
Plots.scatter(xbw[1,:],xbw[2,:],leg=false,border=false, markersize=1)
Plots.savefig("figures/xb_w")
Plots.scatter(sr[1,:],sr[2,:],leg=false,border=false, markersize=1)
Plots.savefig("figures/sb_kgv")

#### Mixing images for poster

using Images, Colors, FixedPointNumbers
using ImageMagick
using TestImages

lena = load("../data/lena.tif")
fabio = load("../data/fabio.tif")
house = load("../data/boat.tif")
