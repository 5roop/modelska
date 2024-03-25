import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import smplotlib
from scipy.integrate import odeint, ode, solve_ivp
from scipy.optimize import root


initial = np.array([0,0,0,0])
def fun(array):
    x,y,z,u = array
    # print(x,y,z,u)
    return [x-1,y-1,z-1, u*z]

sol = root(fun, x0=initial,)
print(sol)