from math import sin, cos, exp


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    else:
        return exp(x) / (1.0 + exp(x))


def sigmoid_d(x):
    if x >= 0:
        return exp(-x) / (exp(-x) + 1) ** 2
    else:
        return exp(x) / (exp(x) + 1) ** 2


def pendulum_ode(
    x,
    t,
    u,
    J=0.000159931461600856,
    m=0.0508581731919534,
    l=0.0415233722862552,
    b=1.43298488358436e-05,
    K=0.0333391179016334,
    R=7.73125142447252,
    c=0.000975041213361349,
    d=165.417960777425,
):
    g = 9.81

    ddx = (u * K / R + m * g * l * sin(x[0]) - b * x[1] - x[1] * K * K / R - c * (2 * sigmoid(d * x[1]) - 1)) / J

    return [x[1], ddx]


def pendulum_dfun(
    x,
    t,
    u,
    J=0.000159931461600856,
    m=0.0508581731919534,
    l=0.0415233722862552,
    b=1.43298488358436e-05,
    K=0.0333391179016334,
    R=7.73125142447252,
    c=0.000975041213361349,
    d=165.417960777425,
):
    g = 9.81

    return [[0, 1], [m * g * l * cos(x[0]) / J, (-b - K * K / R - 2 * c * d * sigmoid_d(d * x[1])) / J]]
