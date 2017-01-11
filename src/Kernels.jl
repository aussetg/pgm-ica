"""
  gauss(x, y[, gamma = 2])

Gaussian Kernel (RBF) in 1-dimension.
"""
function gaussian(x::Float64, y::Float64, gamma = 2)
  return exp(-gamma*(x-y)^2)
end

"""
  laplace(x, y[, gamma = 2])

Gaussian Kernel (RBF) in 1-dimension.
"""
function gauss(x::Float64, y::Float64, gamma = 2)
  return exp(-gamma*abs(x-y))
end

"""
  polynomial(x, y[, d = 2, c = 1])

Gaussian Kernel (RBF) in 1-dimension.
"""
function polynomial(x::Float64, y::Float64, d = 2)
  return (c+x*y)^d
end
