import numpy as np
import matplotlib.pyplot as plt


## Step 1
def load_data(input_path, output_path):
    x = np.loadtxt(input_path)
    y = np.loadtxt(output_path)
    return x, y

x, y = load_data("dataset1_inputs.txt", "dataset1_outputs.txt")

# plt.plot(x, y, "x", color="green")
# plt.show()

## Step 2

mu = np.mean(x)
sigma_sq = np.var(x)
phi = [[] for i in range(len(x))]

for i in range(len(x)):
    for j in range(1, 21):
        phi[i].append(x[i]**(j - 1))


inner = np.dot(np.transpose(phi), phi)
outer = np.dot(np.linalg.inv(inner), np.transpose(phi))
w = np.dot(outer, y)
print w


def compute_fd(x, phi, w):
    data = phi[x]
    total = 0
    for i in range(0, 20):
        total += w[i] * data[i]
    return total


def compute_mse(phi, w):
    mse = np.zeros(20)
    for i in range(0, 20):
        total = 0
        for j in range(len(x)):
            total += (y[j] - compute_fd(j, phi, w))**2
        total /= len(x)
        mse[i] = total
    return mse

sd_vals = compute_mse(phi, w)
plt.plot(range(0, 20), sd_vals)
plt.show()
