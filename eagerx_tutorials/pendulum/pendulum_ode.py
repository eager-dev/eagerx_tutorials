from math import sin


def pendulum_ode(x, t, u, J, m, l, b, K, R):  # noqa:
    g = 9.81

    ddx = (m * g * l * sin(x[0]) - x[1] * (b + K * K / R) + K * u / R) / J

    return [x[1], ddx]
