from scipy.integrate import RK45
import matplotlib.pyplot as plt
import numpy as np
import itertools


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


def ode_system(t, v):
    coeffs = COEFFS_FRANCE

    x, y, z = v

    dinputdt = [
        coeffs[0] * x * (1 - x) + coeffs[1] * x * z / (1 + z) + coeffs[2] * x * y,
        coeffs[3] * y * (1 - y) + coeffs[4] * y * z / (1 + z) + coeffs[5] * x * y,
        coeffs[6] * z * (1 - z) + coeffs[7] * z * x / (1 + x) + coeffs[8] * z * y / (1 + y),
    ]

    return dinputdt


if __name__ == '__main__':
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].remove()
    axs[0, 0] = fig.add_subplot(2, 2, 1, projection='3d')

    x_range = list(range(0, 80, 20))
    y_range = list(range(0, 80, 20))
    z_range = list(range(70, 150, 20))

    X, Y, Z = np.meshgrid(
        np.arange(x_range[0], x_range[-1], 8),
        np.arange(y_range[0], y_range[-1], 8),
        np.arange(z_range[0], z_range[-1], 8),
    )
    coeffs = COEFFS_FRANCE

    Xr = coeffs[0] * X * (1 - X) + coeffs[1] * X * Z / (1 + Z) + coeffs[2] * X * Y
    Yr = coeffs[3] * Y * (1 - Y) + coeffs[4] * Y * Z / (1 + Z) + coeffs[5] * X * Y,
    Zr = coeffs[6] * Z * (1 - Z) + coeffs[7] * Z * X / (1 + X) + coeffs[8] * Z * Y / (1 + Y),

    axs[0, 0].quiver(X, Y, Z, Xr, Yr, Zr, length=0.003)

    v0_list = list(itertools.product(*[x_range, y_range, z_range]))

    for v0 in v0_list:
        solution = RK45(ode_system, 0, v0, 1000, 1)

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

        axs[0, 0].plot(x, y, z, color='r')
        axs[0, 0].plot(v0[0], v0[1], v0[2], marker='o', color='g')
        axs[0, 0].plot(x[-1], y[-1], z[-1], marker='o', color='b')
        print(x[-1], y[-1], z[-1])

        axs[0, 1].plot(x, y, color='r')
        axs[0, 1].plot(v0[0], v0[1], marker='o', color='g')
        axs[0, 1].plot(x[-1], y[-1], marker='o', color='b')

        axs[1, 0].plot(y, z, color='r')
        axs[1, 0].plot(v0[1], v0[2], marker='o', color='g')
        axs[1, 0].plot(y[-1], z[-1], marker='o', color='b')

        axs[1, 1].plot(x, z, color='r')
        axs[1, 1].plot(v0[0], v0[2], marker='o', color='g')
        axs[1, 1].plot(x[-1], z[-1], marker='o', color='b')

    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')

    axs[1, 0].set_xlabel('y')
    axs[1, 0].set_ylabel('z')

    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('z')

    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')
    axs[0, 0].set_zlabel('z')

    plt.show()
