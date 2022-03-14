from scipy.integrate import RK45
import matplotlib.pyplot as plt


COEFFS_FRANCE = [
    .0221,
    .4608,
    -.0077,
    .5585,
    -2.0733,
    .1453,
    .0067,
    1.1813,
    -0.51,
]


COEFFS_ITALY = [
    -.015,
    0.1369,
    -0.1853,
    0.1981,
    -1.1466,
    0.0757,
    0.0013,
    1.0771,
    -1.3245,
]


def ode_system(t, v):
    coeffs = COEFFS_ITALY

    x, y, z = v

    dinputdt = [
        coeffs[0] * x * (1 - x) + coeffs[1] * x * z / (1 + z) + coeffs[2] * x * y,
        coeffs[3] * y * (1 - y) + coeffs[4] * y * z / (1 + z) + coeffs[5] * x * y,
        coeffs[6] * z * (1 - z) + coeffs[7] * z * x / (1 + x) + coeffs[8] * z * y / (1 + y),
    ]

    return dinputdt


if __name__ == '__main__':

    v0 = [25, 3, 115]
    solution = RK45(ode_system, 0, v0, 1000000, 10)

    t_values = []
    y_values = []
    while True:
        # get solution step state
        t_values.append(solution.t)
        y_values.append(solution.y.tolist())
        # break loop after modeling is finished
        if solution.status == 'finished':
            break
        solution.step()

    x, y, z = zip(*y_values)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(x, y, z)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    print(x[-1], y[-1], z[-1])

    plt.show()
