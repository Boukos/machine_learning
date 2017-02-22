import numpy as np
import matplotlib.pyplot as plt
import math


# Step 1
def load_data(input_path, output_path):
    x = np.loadtxt(input_path)
    y = np.loadtxt(output_path)
    return x, y

x, y = load_data("dataset1_inputs.txt", "dataset1_outputs.txt")

# plt.plot(x, y, "x", color="green")
# plt.show()

# Step 2


def poly(i, x):
    return x ** (i - 1)


def generate_phi(D, x):
    phi = [[0] * D for i in range(len(x))]  # initializing multidimensional list
    for i, j in enumerate(x):
        phi[i] = [poly(d, j) for d in range(1, D + 1)]
    return phi


def generate_w_mle(phi, y):
    inner = np.dot(np.transpose(phi), phi)
    outer = np.dot(np.linalg.inv(inner), np.transpose(phi))
    w = np.dot(outer, y)
    return w


def compute_polynomials(w, x, d):
        total = 0
        for i in range(d):
            total += w[i] * poly(i+1, x)
        return total


def compute_mse(D, w, y, x):

    mse = 0
    for j in range(len(y)):
        mse += (y[j] - compute_polynomials(w, x[j], D))**2
    return mse / len(y)


num_basis = 20
mses = np.zeros(num_basis)
for D in range(1, num_basis + 1):
    phi = generate_phi(D, x)
    w = generate_w_mle(phi, y)
    mses[D - 1] = compute_mse(D, w, y, x)

plt.plot(range(0, 20), mses)
plt.title("MSE as a Function of D with MLE Estimate")
plt.xlabel("D")
plt.ylabel("MSE")
plt.xlim(0, 21)
plt.show()


# Step 3
def generate_w_map(phi, y, d):
    lam = 0.001
    w_inner = (lam * np.identity(d) + np.dot(np.transpose(phi), phi))
    w_outer = np.dot(np.linalg.inv(w_inner), np.transpose(phi))
    w_map = np.dot(w_outer, y)
    return w_map

num_basis = 20
mses = np.zeros(num_basis)
for D in range(1, num_basis + 1):
    phi = generate_phi(D, x)
    w = generate_w_map(phi, y, D)
    mses[D - 1] = compute_mse(D, w, y, x)

plt.plot(range(0, 20), mses)
plt.title("MSE as a Function of D with MAP Estimate")
plt.xlabel("D")
plt.ylabel("MSE")
plt.xlim(0, 21)
plt.show()

# Step 4

# y_p_10 = compute_polynomials(w, x, 9)
# y_p_10_map = compute_polynomials(w_map, x, 9)
# # plt.plot(x, y, 'x')
# plt.plot(x, y_p_10, color="red")
# # plt.plot(x, y_p_10_map, color="blue")
# plt.show()

# Step 5

# fold = 10
# avg = np.zeros(20)
# for i in range(0, 20):
#     mses = np.zeros(20)
#     size = 0
#     for j in range(0, len(x), fold):
#         training = np.concatenate([x[0:j], x[j+fold : len(x)]])
#         training_y = np.concatenate([y[0:j], y[j+fold: len(y)]])
#         validation = x[j:j + fold]
#         valid_y = y[j:j + fold]
#         phi = generate_phi(training, 20)
#         w = generate_w_map(phi, training_y)
#         y_prime = compute_polynomials(w, validation, i)
#         mse = 0
#         for k in range(len(validation)):
#             mse += (valid_y[k] - y_prime[k])**2
#         mses[size] = mse/len(y_prime)
#         size += 1
#     avg[i] = np.mean(mses)

# plt.plot(range(0, 20), avg)
# plt.show()

# Step 6

x, y = load_data("dataset2_inputs.txt", "dataset2_outputs.txt")
# plt.plot(x, y, "x", color="green")
# plt.show()

tau = 1000
sigma = 1.5
d = 13
phi = generate_phi(x, d)
inv = (tau**2 * np.identity(d) + sigma**(-2)* np.dot(np.transpose(phi), phi))
cov_w = np.linalg.inv(inv)
# mu_w = sigma**(-2) * np.dot(np.dot(cov_w, np.transpose(phi)), y)
# mu_d = np.dot(np.transpose(mu_w), phi[d])
# # sigma_d = sigma**2 + np.dot(np.dot(np.transpose(x), cov_w), x)
#
# plt.plot(mu_d)
# plt.show()