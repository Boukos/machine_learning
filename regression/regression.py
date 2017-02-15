import numpy as np
import matplotlib.pyplot as plt
import math


## Step 1
def load_data(input_path, output_path):
    x = np.loadtxt(input_path)
    y = np.loadtxt(output_path)
    return x, y

x, y = load_data("dataset1_inputs.txt", "dataset1_outputs.txt")

# plt.plot(x, y, "x", color="green")
# plt.show()

## Step 2


def generate_phi(x):
    phi = [[] for i in range(len(x))]

    for i in range(len(x)):
        for j in range(1, 21):
            phi[i].append(math.pow(x[i], j - 1))
    return phi

phi = generate_phi(x)


def generate_w_mle(phi):
    inner = np.dot(np.transpose(phi), phi)
    outer = np.dot(np.linalg.inv(inner), np.transpose(phi))
    w = np.dot(outer, y)
    return w

w = generate_w_mle(phi)


def compute_fd(j, phi, w):
    data = phi[j]
    new_total = 0
    for i in range(0, 20):
        new_total += w[i] * data[i]
    return new_total


def compute_mse(phi, w, y):
    mse = np.zeros(20)
    for i in range(0, 20):
        total = 0
        for j in range(len(x)):
            total += (y[j] - compute_fd(j, phi, w))**2
        total /= len(phi)
        mse[i] = total
    return mse

sd_vals = compute_mse(phi, w, y)
# plt.plot(range(0, 20), sd_vals)
# plt.show()


## Step 3
def generate_w_map(phi):
    lam = 0.001
    w_inner = (lam * np.identity(20) + np.dot(np.transpose(phi), phi))
    w_outer = np.dot(np.linalg.inv(w_inner), np.transpose(phi))
    w_map = np.dot(w_outer, y)
    return w_map

w_map = generate_w_map(phi)
lam_sd_vals = compute_mse(phi, w_map, y)
# plt.plot(range(0, 20), lam_sd_vals)
# plt.show()


## Step 5

fold = 10
for i in range(0, 20):
    for j in range(0, len(x), fold):
        training = np.concatenate([x[0:j], x[j+fold : len(x)]])
        validation = x[j:j + fold]
        phi = generate_phi(training)
        w = generate_w_map(phi)
        mse = compute_mse(phi, w, y)

