import numpy as np
import matplotlib.pyplot as plt

N, p = 30, 20
np.random.seed(0)
X = np.random.randn(N, p)
Y = 2 * np.random.randint(2, size=N) - 1
theta = np.random.randn(p)


def loss():
    return np.sum(np.maximum(0, 1 - Y * np.dot(X, theta))) / N + 0.1 * np.dot(
        theta, theta
    )


non_diff = []


def stochastic_gradient(i):
    if Y[i] * np.dot(X[i], theta) == 1:
        non_diff.append(i)
    return -Y[i] * X[i] / N if (1 - Y[i] * np.dot(X[i], theta)) >= 0 else np.zeros(p)


result = []
for _ in range(20000):
    result.append(loss())
    i = np.random.randint(N)
    theta -= 0.05 * stochastic_gradient(i)

plt.plot(result)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.title("Problem 2(SVM-SGD)")
plt.savefig("Problem 2.pdf")

print(non_diff)
