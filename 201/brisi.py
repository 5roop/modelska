# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint, ode, solve_ivp

# plt.style.use("ggplot")
# plt.rcParams['font.family'] = "serif"
# plt.rcParams['font.size'] = 13
# plt.rcParams.update({'figure.autolayout': True})
# plt.rc('text', usetex=True)

import smplotlib

pd.set_option("display.max_columns", 50)


def odes(t, arej):
    x, y, u, v = arej
    xpika = u
    ypika = v
    upika = -x * np.power(x**2 + y**2, -3 / 2)
    vpika = -y * np.power(x**2 + y**2, -3 / 2)

    return [xpika, ypika, upika, vpika]





def calculate_H(sol):
    def _H(arej):
        x, y, u, v = arej
        return 0.5 * (u**2 + v**2) + 1 / (x**2 + y**2)

    return np.apply_along_axis(_H, 1, sol)


def calculate_L(sol):
    def _absL(arej):
        x, y, u, v = arej
        return np.linalg.norm(np.cross([x, y, 0], [u, v, 0]))

    return np.apply_along_axis(_absL, 1, sol)


def calculate_A(sol):
    def _absA(arej):
        x, y, u, v = arej
        return np.linalg.norm(
            np.cross([x, y, 0], [u, v, 0])
            - np.array([x, y, 0]) / np.linalg.norm([x, y, 0])
        )

    return np.apply_along_axis(_absA, 1, sol)


v0s = [1, np.sqrt(2), 1.5, 0.5, 0.3]
rezs = []
for v0 in v0s:
    initial = [1, 0, 0, v0]
    dt = 0.01
    t_span = np.arange(0, 10, dt)

    sol = solve_ivp(
        odes,
        t_span=[t_span.min(), t_span.max()],
        t_eval=t_span,
        tfirst=True,
        y0=initial,
        dense_output=True,
        rtol=1e-20,
        atol=1e-20,
    )
    sol.H = calculate_H(sol.y.T)
    sol.L = calculate_L(sol.y.T)
    sol.A = calculate_A(sol.y.T)
    rezs.append(sol)

# %%
for sol in rezs:
    v0 = sol.y[3, 0]
    if v0 == np.sqrt(2):
        v0 = "\sqrt{2}"
    else:
        v0 = f"{v0:0.1f}"
    plt.plot(sol.y[0, :], sol.y[1, :], label=rf"$v_0={v0}$")
plt.legend()
plt.savefig("images/1-1-v0.pdf")

# %%
for sol in rezs:
    v0 = sol.y[3, 0]
    if v0 == np.sqrt(2):
        v0 = "\sqrt{2}"
    else:
        v0 = f"{v0:0.1f}"
    Q, strQ = sol.H, "H"
    plt.plot(
        sol.t,  # np.log10(np.abs((Q[0] - Q) )),
        np.log10(np.abs(Q[0] - Q)),
        label=rf"$v_0={v0}$",
    )
plt.ylabel(rf"$ \log10 |{strQ}_0 - {strQ}|$")
plt.legend()
plt.title(f"${strQ}$")
plt.savefig(f"images/1-1-{strQ}_log.pdf")
plt.show()
plt.clf()
plt.clf()

for sol in rezs:
    v0 = sol.y[3, 0]
    if v0 == np.sqrt(2):
        v0 = "\sqrt{2}"
    else:
        v0 = f"{v0:0.1f}"
    Q, strQ = sol.H, "H"
    plt.plot(
        sol.t,  # np.log10(np.abs((Q[0] - Q) )),
        np.abs(Q[0] - Q),
        label=rf"$v_0={v0}$",
    )
plt.ylabel(rf"$|{strQ}_0 - {strQ}|$")
plt.ylim((None, 100))
plt.legend()
plt.title(f"${strQ}$")
plt.savefig(f"images/1-1-{strQ}_lin.pdf")
plt.show()
plt.clf()

for sol in rezs:
    v0 = sol.y[3, 0]
    if v0 == np.sqrt(2):
        v0 = "\sqrt{2}"
    else:
        v0 = f"{v0:0.1f}"
    Q, strQ = sol.L, "L"
    plt.plot(
        sol.t,  # np.log10(np.abs((Q[0] - Q) )),
        np.log10(np.abs(Q[0] - Q)),
        label=rf"$v_0={v0}$",
    )
plt.ylabel(rf"$ \log10 |{strQ}_0 - {strQ}|$")
plt.legend()
plt.title(f"${strQ}$")
plt.savefig(f"images/1-1-{strQ}_log.pdf")
plt.show()
plt.clf()
plt.clf()

for sol in rezs:
    v0 = sol.y[3, 0]
    if v0 == np.sqrt(2):
        v0 = "\sqrt{2}"
    else:
        v0 = f"{v0:0.1f}"
    Q, strQ = sol.L, "L"
    plt.plot(
        sol.t,  # np.log10(np.abs((Q[0] - Q) )),
        np.abs(Q[0] - Q),
        label=rf"$v_0={v0}$",
    )
plt.ylabel(rf"$|{strQ}_0 - {strQ}|$")
plt.legend()
plt.title(f"${strQ}$")
plt.savefig(f"images/1-1-{strQ}_lin.pdf")
plt.show()
plt.clf()


for sol in rezs:
    v0 = sol.y[3, 0]
    if v0 == np.sqrt(2):
        v0 = "\sqrt{2}"
    else:
        v0 = f"{v0:0.1f}"
    Q, strQ = sol.A, "A"
    plt.plot(
        sol.t,  # np.log10(np.abs((Q[0] - Q) )),
        np.log10(np.abs(Q[0] - Q)),
        label=rf"$v_0={v0}$",
    )
plt.ylabel(rf"$ \log10 |{strQ}_0 - {strQ}|$")
plt.legend()
plt.title(f"${strQ}$")
plt.savefig(f"images/1-1-{strQ}_log.pdf")
plt.show()
plt.clf()
plt.clf()

for sol in rezs:
    v0 = sol.y[3, 0]
    if v0 == np.sqrt(2):
        v0 = "\sqrt{2}"
    else:
        v0 = f"{v0:0.1f}"
    Q, strQ = sol.A, "A"
    plt.plot(
        sol.t,  # np.log10(np.abs((Q[0] - Q) )),
        np.abs(Q[0] - Q),
        label=rf"$v_0={v0}$",
    )
plt.ylabel(rf"$|{strQ}_0 - {strQ}|$")
plt.legend()
plt.title(f"${strQ}$")
plt.savefig(f"images/1-1-{strQ}_lin.pdf")
plt.show()
plt.clf()

# %%
v0 = 1
initial = [1, 0, 0, v0]
dt = 1.0
t_span = np.arange(0, 15, dt)


def camearound(t, y):
    return y[1]


camearound.terminal = False
camearound.direction = 1

sol = solve_ivp(
    odes,
    t_span=[0, 60],
    y0=initial,
    dense_output=True,
    events=camearound,
    rtol=1e-20,
    atol=1e-20,
)

sol.t_events[0] / np.pi

# %%
sols = []
v0s = np.linspace(0.05, np.sqrt(2), 20)
for v0 in v0s:
    initial = [1, 0, 0, v0]
    if v0 == np.sqrt(2):
        v0str = "\sqrt{2}"
    else:
        v0str = f"{v0:0.1f}"

    def camearound(t, y):
        return y[1]

    camearound.terminal = False
    camearound.direction = 1

    sol = solve_ivp(
        odes,
        t_span=[0, 21 * np.pi],
        y0=initial,
        dense_output=True,
        events=camearound,
        rtol=1e-20,
        atol=1e-20,
    )
    sol.v0 = v0
    sol.v0str = v0str
    sols.append(sol)
import seaborn as sns
import pandas as pd

df = pd.DataFrame(
    data={
        "v0": [sol.v0 for sol in sols],
        "v0str": [sol.v0str for sol in sols],
        "tavg": [np.mean(np.diff(sol.t_events, 1)) for sol in sols],
        "tstd": [np.std(np.diff(sol.t_events, 1)) for sol in sols],
    }
)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axes[0].errorbar(df.v0, df.tavg, yerr=df.tstd, color="k")
axes[0].set_ylabel("$t_0$")
axes[0].set_xlabel("$v_0$")
axes[1].scatter(df.v0, df.tstd, color="k")
axes[1].set_xlabel("$v_0$")
axes[1].set_ylabel("$\sigma(t_0)$")
fig.tight_layout()
fig.savefig("images/1-2-t0.pdf")
plt.show()
df

# %%


# %%


# %%


# %%
r.integrate(10, 0.001)

# %%
t, sol = integrate()
fig, ax = plt.subplots()
ax.plot(sol[:, 0], sol[:, 1])
ax.set_aspect("equal")
plt.savefig("0-simple-trajectory.pdf")
plt.show()


# %%
def integrate(
    t_max: "Time until the end of integration" = 10,
    dt: "Time step" = 0.1,
    initial: "Array of initial conditions [x, y, xdot, ydot]" = [1, 0, 0, 1],
):
    def odes(t, arej):
        x, y, u, v = arej
        xpika = u
        ypika = v
        upika = -x * np.power(x**2 + y**2, -3 / 2)
        vpika = -y * np.power(x**2 + y**2, -3 / 2)

        return [xpika, ypika, upika, vpika]

    t = np.arange(0, t_max, dt)
    r = ode(odes).set_integrator("dopri5")
    r.set_initial_value(initial, t[0])
    sol = np.zeros((t.shape[0], 4))  # Result buffer
    for i, t_i in enumerate(t):
        sol[i, :] = r.integrate(t_i)
    return t, sol


def plot_trajectory(sol: "ndarray with columns x, y, xdot, ydot"):
    initial_conditions = sol[0, :]
    fig, ax = plt.subplots()
    ax.plot(sol[:, 0], sol[:, 1])
    ax.set_aspect("equal")
    plt.show()


def calculate_H(sol):
    def _H(arej):
        x, y, u, v = arej
        return 0.5 * (u**2 + v**2) + 1 / (x**2 + y**2)

    return np.apply_along_axis(_H, 1, sol)


def calculate_L(sol):
    def _absL(arej):
        x, y, u, v = arej
        return np.linalg.norm(np.cross([x, y, 0], [u, v, 0]))

    return np.apply_along_axis(_absL, 1, sol)


def calculate_A(sol):
    def _absA(arej):
        x, y, u, v = arej
        return np.linalg.norm(
            np.cross([x, y, 0], [u, v, 0])
            - np.array([x, y, 0]) / np.linalg.norm([x, y, 0])
        )

    return np.apply_along_axis(_absA, 1, sol)


# %%
dt = 0.1
t, sol = integrate(dt=dt)
plt.plot(t, calculate_H(sol), label=f"dt = {dt}")
dt = 0.08
t, sol = integrate(dt=dt)
plt.plot(t, calculate_H(sol), label=f"dt = {dt}")
dt = 0.06
t, sol = integrate(dt=dt)
plt.plot(t, calculate_H(sol), label=f"dt = {dt}")
dt = 0.04
t, sol = integrate(dt=dt)
plt.plot(t, calculate_H(sol), label=f"dt = {dt}")
plt.legend()
plt.title("""$\mathcal{H}$ pri različnih časovnih korakih""", pad=20)
plt.ylabel("$\mathcal{H}$ (arbitrarne enote)")
plt.xlabel("čas (arbitrarne enote)")
plt.savefig("1-H.pdf")

# %%
dts = np.logspace(-4, 0, 10)
Hs = np.empty_like(dts)
Ls = np.empty_like(dts)
As = np.empty_like(dts)
for i, dt in enumerate(dts):
    print(f"i={i}", end="\r")
    sol = integrate(dt=dt)[1]
    Hs[i] = np.std(calculate_H(sol))
    Ls[i] = np.std(calculate_L(sol))
    As[i] = np.std(calculate_A(sol))

# %%
plt.figure(figsize=(5, 4))
plt.scatter(dts, Hs, label=r"$\mathcal{H}$")
plt.scatter(dts, Ls, label=r"$\|\vec{l} \|$")
plt.scatter(dts, As, label=r"$\|\vec{A} \|$")
plt.semilogy()
plt.semilogx()
plt.xlim(0.5e-4, 2)
plt.ylim(4e-16, 2e-5)
plt.xlabel("Časovni korak")
plt.ylabel("Standardni odklon invariantnih količin")
plt.legend()
plt.savefig("1-stds.pdf")


# %%
def simple_poincare(sol, dt):
    x, y, u, v = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]
    rezs = []
    for i, current_x in enumerate(x):
        if i == 0:
            continue
        if (x[i - 1] > 0) and (current_x < 0):
            alpha = np.abs(x[i - 1]) / dt
            rezs.append(
                [
                    alpha * y[i - 1] + (1 - alpha) * y[i],
                    alpha * v[i - 1] + (1 - alpha) * v[i],
                ]
            )
    return np.array(rezs)


dt = 1
t, sol = integrate(dt=dt, t_max=30000)
poincare = simple_poincare(sol, dt)

fig, [ax1, ax2] = plt.subplots(ncols=2)
ax1.plot(sol[:, 0], sol[:, 1], label=f"dt={dt}")
ax1.set_aspect("equal")
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax1.set_title("Trajektorija")
ax2.scatter(poincare[:, 0], poincare[:, 1], label=f"dt={dt}")
ax2.set_title("Poincaréjev presek")
ax2.set_aspect("equal")

dt = 0.1
t, sol = integrate(dt=dt, t_max=30000)
poincare = simple_poincare(sol, dt)
ax1.plot(sol[:, 0], sol[:, 1], label=f"dt={dt}")
ax2.scatter(poincare[:, 0], poincare[:, 1], label=f"dt={dt}")
ax1.legend()
ax2.legend()
plt.savefig("1-poincare1-1.png", dpi=300)

# %%
fig, [ax1, ax2] = plt.subplots(ncols=2)
ax1.set_aspect("equal")
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax1.set_title("Trajektorija")
ax2.set_title("Poincaréjev presek")
# ax2.set_aspect("equal")

dt = 0.04
t, sol = integrate(dt=dt, t_max=1000, initial=[1, 0, 0, 1])
poincare = simple_poincare(sol, dt)
ax1.plot(sol[:, 0], sol[:, 1], label=f"$v_0=1$")
ax2.scatter(poincare[:, 0], poincare[:, 1], label=f"$v_0=1$")

t, sol = integrate(dt=dt, t_max=1000, initial=[1, 0, 0, 0.5])
poincare = simple_poincare(sol, dt)
ax1.plot(sol[:, 0], sol[:, 1], label=f"$v_0=.5$")
ax2.scatter(poincare[:, 0], poincare[:, 1], label=f"$v_0=.5$")

t, sol = integrate(dt=dt, t_max=1000, initial=[1, 0, 0, 0.3])
poincare = simple_poincare(sol, dt)
ax1.plot(sol[:, 0], sol[:, 1], label=f"$v_0=.3$")
ax2.scatter(poincare[:, 0], poincare[:, 1], label=f"$v_0=.3$")

ax1.legend()
ax2.legend()
# plt.savefig("1-poincare4.pdf")

# %%
a = np.arange(0, 10, 1)
np.diff(a)


# %%
def simple_periods(sol, t, dt):
    x, y, u, v = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]
    rezs = []
    for i, current_x in enumerate(x):
        if i == 0:
            continue
        if (x[i - 1] > 0) and (current_x < 0):
            alpha = np.abs(x[i - 1]) / dt
            rezs.append(alpha * t[i - 1] + (1 - alpha) * t[i])
    return np.array(rezs)


t_mean = []
t_err = []
dts = np.logspace(-2.5, 0, 7)
for dt in dts:
    t, sol = integrate(dt=dt, t_max=1000, initial=[1, 0, 0, 1])
    periods = simple_periods(sol, t, dt)
    t_mean.append(np.mean(np.diff(periods)))
    t_err.append(np.std(np.diff(periods)))


# %%
fig, ax = plt.subplots()
ax.errorbar(dts, t_mean, yerr=t_err, fmt="or")
ax.set_xscale("log")
ax.set_ylabel("Obhodni čas")
ax.set_xlabel("Časovni korak")
plt.savefig("1-periode.pdf")
plt.show()

# %%
t_mean = []
t_err = []
dts = np.logspace(-2.5, 0, 7)
for dt in dts:
    t, sol = integrate(dt=dt, t_max=1000, initial=[1, 0, 0, 0.3])
    periods = simple_periods(sol, t, dt)
    t_mean.append(np.mean(np.diff(periods)))
    t_err.append(np.std(np.diff(periods)))
fig, ax = plt.subplots()
ax.errorbar(dts, t_mean, yerr=t_err, fmt="or")
ax.set_xscale("log")
ax.set_ylabel("Obhodni čas")
ax.set_xlabel("Časovni korak")
plt.savefig("1-periode_elipticno.pdf")
plt.show()

# %%
fig, ax = plt.subplots()
ax.errorbar(dts, t_mean, yerr=t_err, fmt="or")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("Obhodni čas")
ax.set_xlabel("Časovni korak")
plt.savefig("1-periode_elipticno.pdf")
plt.show()

# %% [markdown]
# # Druga podnaloga


# %%
def integrate2(
    t_max: "Time until the end of integration" = 10,
    dt: "Time step" = 0.1,
    initial: "Array of initial conditions [x, y, xdot, ydot]" = [1, 0, 0, 1],
    R: "Ratio of intrastellar radius to planetary radius" = 0.5,
    faza: "Offset of the binary constelation relative to y=0 line" = 0,
):
    def odes(t, arej, faza=faza):
        x, y, u, v = arej
        xpika = u
        ypika = v
        x1 = R * np.cos(t + faza)
        y1 = R * np.sin(t + faza)
        x2 = R * np.cos(t + np.pi + faza)
        y2 = R * np.sin(t + np.pi + faza)
        upika = -0.5 * (x - x1) * np.power(
            (x - x1) ** 2 + (y - y1) ** 2, -3 / 2
        ) - 0.5 * (x - x2) * np.power((x - x2) ** 2 + (y - y2) ** 2, -3 / 2)
        vpika = -0.5 * (y - y1) * np.power(
            (x - x1) ** 2 + (y - y1) ** 2, -3 / 2
        ) - 0.5 * (y - y2) * np.power((x - x2) ** 2 + (y - y2) ** 2, -3 / 2)
        return [xpika, ypika, upika, vpika]

    t = np.arange(0, t_max, dt)
    r = ode(odes).set_integrator("dopri5")
    r.set_initial_value(initial, t[0])
    sol = np.zeros((t.shape[0], 4))  # Result buffer
    for i, t_i in enumerate(t):
        sol[i, :] = r.integrate(t_i)
    return t, sol


def circle(R):
    fi = np.linspace(0, 2 * np.pi, 100)
    x = R * np.cos(fi)
    y = R * np.sin(fi)
    return x, y


R = 0.1
dt = 0.01
t, sol = integrate2(dt=dt, R=R, t_max=200)
fig, [ax1, ax2] = plt.subplots(ncols=2)
ax1.set_aspect("equal")
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax1.set_title("Trajektorija")
ax2.set_title("Poincaréjev presek")
ax2.set_aspect("equal")
poincare = simple_poincare(sol, dt)
ax1.plot(
    sol[:, 0],
    sol[:, 1],
)
x, y = circle(R)
ax1.plot(x, y)
ax2.scatter(
    poincare[:, 0],
    poincare[:, 1],
)
# plt.savefig("2-small_R.pdf")
plt.show()

# %%
R = 0.5
dt = 0.01
t, sol = integrate2(dt=dt, R=R, t_max=200)
fig, [ax1, ax2] = plt.subplots(ncols=2)
ax1.set_aspect("equal")
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax1.set_title("Trajektorija")
ax2.set_title("Poincaréjev presek")
ax2.set_aspect("equal")
poincare = simple_poincare(sol, dt)
ax1.plot(
    sol[:, 0],
    sol[:, 1],
)
ax2.scatter(
    poincare[:, 0],
    poincare[:, 1],
)
x, y = circle(R)
ax1.plot(x, y)
plt.savefig("2-big_R.pdf")
plt.show()

# %%
R = 1
dt = 0.01
t, sol = integrate2(dt=dt, R=R, t_max=5, faza=0.2)
fig, [ax1, ax2] = plt.subplots(ncols=2)
ax1.set_aspect("equal")
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax1.set_title("Trajektorija")
ax2.set_title("Poincaréjev presek")
# ax2.set_aspect("equal")
poincare = simple_poincare(sol, dt)
ax1.plot(
    sol[:, 0],
    sol[:, 1],
)
ax2.scatter(
    poincare[:, 0],
    poincare[:, 1],
)
x, y = circle(R)
ax1.plot(x, y)
# plt.savefig("2-bigbig_R.pdf")
plt.show()


# %%
def integrate3(
    t_max: "Time until the end of integration" = 10,
    dt: "Time step" = 0.1,
    initial: "Array of initial conditions [x, y, xdot, ydot]" = [1, 0, 0, 1],
    faza: "Offset of the planet" = 0,
):
    def odes(t, arej, faza=faza):
        x, y, u, v = arej
        xpika = u
        ypika = v
        x1 = 0
        y1 = 0
        x2 = -10 + 2 * t
        y2 = 1.5
        upika = -(x - x1) * np.power((x - x1) ** 2 + (y - y1) ** 2, -3 / 2) - (
            x - x2
        ) * np.power((x - x2) ** 2 + (y - y2) ** 2, -3 / 2)
        vpika = -(y - y1) * np.power((x - x1) ** 2 + (y - y1) ** 2, -3 / 2) - (
            y - y2
        ) * np.power((x - x2) ** 2 + (y - y2) ** 2, -3 / 2)
        return [xpika, ypika, upika, vpika]

    t = np.arange(0, t_max, dt)
    r = ode(odes).set_integrator("dopri5")
    r.set_initial_value(initial, t[0])
    sol = np.zeros((t.shape[0], 4))  # Result buffer
    for i, t_i in enumerate(t):
        sol[i, :] = r.integrate(t_i)
    return t, sol


def simple_poincare(sol, dt):
    x, y, u, v = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]
    rezs = []
    for i, current_x in enumerate(x):
        if i == 0:
            continue
        if (x[i - 1] > 0) and (current_x < 0):
            alpha = np.abs(x[i - 1]) / dt
            rezs.append(
                [
                    alpha * y[i - 1] + (1 - alpha) * y[i],
                    alpha * v[i - 1] + (1 - alpha) * v[i],
                ]
            )
    return np.array(rezs)


# %%
dt = 1
t, sol = integrate3(dt=dt, t_max=800)
fig, [ax1, ax2] = plt.subplots(ncols=2)
ax1.set_aspect("equal")
# ax1.set_yticklabels([])
# ax1.set_xticklabels([])
ax1.set_title("Trajektorija")
poincare = simple_poincare(sol, dt)
ax1.plot(
    sol[:, 0],
    sol[:, 1],
)
ax2.scatter(
    poincare[:, 0],
    poincare[:, 1],
)
ax2.set_title("Poincaréjev presek")
ax2.set_aspect("equal")
plt.savefig("3-trajektorija.pdf")
plt.show()

# %%
poincare

# %%
