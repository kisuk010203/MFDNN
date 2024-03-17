import numpy as np
import matplotlib.pyplot as plt


N = 30
np.random.seed(0)
X = np.random.randn(2, N)
y = np.sign(X[0, :] ** 2 + X[1, :] ** 2 - 0.7)
theta = 0.5
c, s = np.cos(theta), np.sin(theta)
X = np.array([[c, -s], [s, c]]) @ X
X = X + np.array([[1], [1]])


w = [1.3, -2, 1, -2, 1]
xx = np.linspace(-4, 4, 1024)
yy = np.linspace(-4, 4, 1024)
xx, yy = np.meshgrid(xx, yy)
Z = w[0] + (w[1] * xx + w[2] * xx**2) + (w[3] * yy + w[4] * yy**2)

# plt.xlabel("u")
# plt.ylabel("v")
# plt.title("Original Decision Boundary")
# plt.plot(X[0, y == 1], X[1, y == 1], "ro", label="1")
# plt.plot(X[0, y == -1], X[1, y == -1], "bo", label="-1")
# plt.contour(xx, y, Z, 0)
# plt.legend()
# plt.savefig("Problem 3.pdf")


def loss():
    return np.sum(np.maximum(0, 1 - y * (np.dot(X, theta)))) / N + 0.1 * np.dot(
        theta, theta
    )


def stochastic_gradient(i):
    return -y[i] * X[i] / N if (1 - y[i] * np.dot(X[i], theta)) >= 0 else np.zeros(5)


u, v = X[0], X[1]
X = np.vstack((np.ones(u.shape[0]), u, u**2, v, v**2)).T
theta = np.random.randn(5)
for _ in range(50000):
    i = np.random.randint(N)
    theta -= 0.05 * stochastic_gradient(i)

w = theta
xx = np.linspace(-4, 4, 1024)
yy = np.linspace(-4, 4, 1024)
xx, yy = np.meshgrid(xx, yy)
Z = w[0] + (w[1] * xx + w[2] * xx**2) + (w[3] * yy + w[4] * yy**2)

plt.xlabel("u")
plt.ylabel("v")
plt.title("Trained Linear Classifier by SVM-SGD")
plt.plot(X[y == 1, 1], X[y == 1, 3], "ro", label="1")
plt.plot(X[y == -1, 1], X[y == -1, 3], "bo", label="-1")
plt.contour(xx, yy, Z, 0)
plt.legend()
# plt.show()
plt.savefig("Problem 3_2.pdf")
