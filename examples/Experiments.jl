include("../src/FastICA.jl")

using Distributions
import Plots
Plots.gr()

u = Uniform(5,10)

s1, s2 = rand(u,Int64(1e4)), rand(u,Int64(1e4))

s = [s1 s2]

Plots.scatter(s[:,1],s[:,2],leg=false,border=false, markersize=0.8)
Plots.savefig("figures/s_orig")

A = [1 2; 21 20]

x = (A * s')

Plots.scatter(x[1,:],x[2,:],leg=false,border=false, markersize=0.8)
Plots.savefig("figures/x")

xc , m = centeringt(x)

Plots.scatter(xc[1,:],xc[2,:],leg=false,border=false, markersize=0.8)
Plots.savefig("figures/x_c")

xw = whitent!(xc)

Plots.scatter(xw[1,:],xw[2,:],leg=false,border=false, markersize=0.8)
Plots.savefig("figures/x_w")

w, s_ica = fastICA(xw, 2, 50, 1e-6)

Plots.scatter(s_ica[1,:],s_ica[2,:],leg=false,border=false, markersize=0.8)
Plots.savefig("figures/s_retrieved")

#### A more interesting Distribution

include("../src/KernelICA.jl")

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

xbc, mb = centeringt(xb)
xbw = whitent!(xbc)

w_ica, sf = fastICA(xbw, 2)
w_kica, sr = kica(xbw)

Plots.scatter(sf[1,:],sf[2,:],leg=false,border=false, markersize=1)
Plots.savefig("figures/sb_ica_orig")

Plots.scatter(sb[1,:],sb[2,:],leg=false,border=false, markersize=1)
Plots.savefig("figures/sb_orig")
Plots.scatter(xb[1,:],xb[2,:],leg=false,border=false, markersize=1)
Plots.savefig("figures/xb")
Plots.scatter(xbw[1,:],xbw[2,:],leg=false,border=false, markersize=1)
Plots.savefig("figures/xb_w")
Plots.scatter(sf[1,:],sf[2,:],leg=false,border=false, markersize=1)
Plots.savefig("figures/sb_ica")
Plots.scatter(sr[1,:],sr[2,:],leg=false,border=false, markersize=1)
Plots.savefig("figures/sb_kgv")
