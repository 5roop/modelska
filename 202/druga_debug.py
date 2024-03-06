import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import smplotlib
from scipy.integrate import odeint, ode, solve_ivp
from scipy.optimize import root


initial = np.array([0, 0, -0.1, 0.12])


def U(x, y):
    return 0.5 * x**2 + 0.5 * (y**2) + (x**2) * y - (y**3) / 3


def fun_to_minimize(initial):
    def odes(t, INP):
        """The main set of equations"""
        Y = np.zeros(4)
        x, y, u, v = INP
        Y[0] = u
        Y[1] = v
        Y[2] = -x - 2 * x * y
        Y[3] = -y - x**2 + y**2
        return Y  # For odeint

    x, y, v, u = initial
    assert U(x, y) + 0.5 * v**2 + 0.5 * u**2 < 1 / 6

    t_span = np.linspace(0, 100, 100)

    def crosses_x(t, arej):
        return arej[1]

    crosses_x.direction = 1
    crosses_x.terminal = False

    sol = solve_ivp(
        odes,
        t_span=[t_span.min(), t_span.max()],
        t_eval=t_span,
        tfirst=True,
        y0=initial,
        dense_output=True,
        rtol=1e-20,
        atol=1e-20,
        events=crosses_x,
    )
    take_idx = 0 if sol.t_events[0][0] > 0 else 1
    end_point = sol.y_events[0][take_idx]
    # print(end_point, type(end_point))
    return initial - end_point, sol


bla, sol = fun_to_minimize(initial)
