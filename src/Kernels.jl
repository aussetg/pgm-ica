"""
  gaussian(x, y[, gamma = 2])

Gaussian Kernel (RBF) in 1-dimension.
"""
@fastmath function gaussian(x::Float64, y::Float64, gamma = 2)
  return exp(-gamma*(x-y)^2)
end

"""
  laplace(x, y[, gamma = 2])

Gaussian Kernel (RBF) in 1-dimension.
"""
@fastmath function laplace(x::Float64, y::Float64, gamma = 2)
  return exp(-gamma*abs(x-y))
end

"""
  polynomial(x, y[, d = 2, c = 1])

Gaussian Kernel (RBF) in 1-dimension.
"""
@fastmath function polynomial(x::Float64, y::Float64, d = 2, c = 1)
  return (c+x*y)^d
end
