import numpy as np

def gradient_descent(start, gradient, learn_rate, max_iter, tol=0.01):
  steps = [start] # history tracking
  x = start
  
  decay_rate = 0.00005
  
  
  
  for i in range(max_iter):
    learn_rate = learn_rate/(1 + decay_rate * i-1) # adjust learn_rate
    diff = learn_rate*gradient(x)
    if np.abs(diff)<tol:
      break    
    x = x - diff
    steps.append(x) # history tracing

  return steps, x
