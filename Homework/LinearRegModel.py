import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([35., 38., 31., 20., 22., 25., 17., 60., 8., 60.])
y_data = 2 * x_data + 50 + 5 * np.random.random()
initial_b = 0  # bias
initial_w = 0  # weight
b_history = [initial_b]  # bias history
w_history = [initial_w]  # weight history
learning_rate = 0.0005
num_iterations = 15000


def run():
    """
    Main run method: initiates gradient descent, plots gradient, and
    compares the real and predicted y_value.
    """

    [b, w] = gradient_descent(x_data, initial_b, initial_w, learning_rate, num_iterations)
    bb = np.arange(0, 100, 1)  # bias
    ww = np.arange(-5, 5, 0.1)  # weight
    Z = np.zeros((len(bb), len(ww)))

    for i in range(len(bb)):
        for j in range(len(ww)):
            b = bb[i]
            w = ww[j]
            Z[j][i] = 0
            for n in range(len(x_data)):
                Z[j][i] = Z[j][i] + (w * x_data[n] + b - y_data[n]) ** 2
            Z[j][i] = Z[j][i] / len(x_data)

    plt.xlim(0, 100)
    plt.ylim(-5, 5)
    plt.contourf(bb, ww, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
    plt.plot([50], [2], 'x', ms=12, markeredgewidth=3, color='orange')
    plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
    plt.show()      # plots the gradient step history

    for z in range(0, len(x_data)):
        y_real = 2 * x_data[z] + 50
        y_pred = 2 * x_data[z] + 50 + 5 * np.random.random()
        plt.plot(x_data[z], y_real, 'x', color='red')
        plt.plot(x_data[z], y_pred, '+', color='blue')

    plt.show()      # plots the comparison of real/pred. y_values


def gradient_descent(x_data, starting_b, starting_w, learning_rate, num_iterations):
    """
    Gradually steps through gradient based on the number of iterations or an acceptable error.
    returns: [bias, weight]
    """

    b = starting_b
    w = starting_w

    for i in range(num_iterations):
        b, w = step_gradient(b, w, x_data, learning_rate)
        b_history.append(b)  # stores bias approximations to plot
        w_history.append(w)  # stores weight approximations to plot
        err = error(b, w, x_data)
        if err <= .6:  # if the error is acceptable exit iterations loop
            print('error = % f' % err)
            break
    return [b, w]


def error(b, m, x_data):
    """
    Method to compute the accuracy of our step_gradient's approximation
    returns: totalError
    """

    totalError = 0
    for i in range(0, len(x_data)):
        x = x_data[i]
        y = 2 * x_data[i] + 50 + 5 * np.random.random()

        totalError += (y - (m * x + b)) ** 2        # total error of gradient

    return totalError / float(len(x_data))


def step_gradient(b_current, w_current, x_data, learning_rate):
    """
    Method to compute the approximate gradient. Returns a new bias and a new weight
    returns: [new_bias, new_weight]
    """

    b_gradient = 0
    w_gradient = 0
    N = float(len(x_data))

    for i in range(0, len(x_data)):
        x = x_data[i]
        y = 2 * x_data[i] + 50 + 5 * np.random.random()

        b_gradient += -(2 / N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2 / N) * x * (y - ((w_current * x) + b_current))

    new_b = b_current - (learning_rate * b_gradient)
    new_w = w_current - (learning_rate * w_gradient)

    return [new_b, new_w]


if __name__ == '__main__':
    run()   # run process
