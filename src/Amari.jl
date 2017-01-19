"""
  amari(w1, w2)

Compute the Amari distance between w1 and w2. They need to be square. This distance function is between 0 and 1 and is invariant to scaling and permutations.
See: Amari, S., Cichocki, A., & Yang, H. H. (1996). A New Learning Algorithm for Blind Signal Separation. Retrieved from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.7.6283&rep=rep1&type=pdf
"""

function amari(w1, w2)
  m = size(w1,1)
  P = abs(w1^(-1) * w2)
  d = mean(sum(P,1) ./ mapslices(maximum, P, 1) - 1) + mean(sum(P,2) ./ mapslices(maximum, P, 2) - 1)
  return 0.5*d
end
