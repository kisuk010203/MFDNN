import numpy as np
import matplotlib.pyplot as plt

N, p = 30, 20
np.random.seed(0)
X = np.random.randn(N, p)
Y = 2 * np.random.randint(2, size=N) - 1
theta = np.zeros(p)


def loss():
    return np.sum(np.log(1 + np.exp(-Y * np.dot(X, theta)))) / N


def stochastic_gradient(i):
    return -Y[i] * X[i] / ((1 + np.exp(Y[i] * np.dot(X[i], theta))) * N)


result = []
for _ in range(200000):
    result.append(loss())
    i = np.random.randint(N)
    theta -= 0.2 * stochastic_gradient(i)

plt.plot(result)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.title("Problem 1(SGD)")
plt.savefig("Problem 1.pdf")
