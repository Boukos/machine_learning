import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.interpolate import spline

# Step 1
def load_data(input_path, output_path):
    x = np.loadtxt(input_path)
    y = np.loadtxt(output_path)
    return x, y

x, y = load_data("dataset1_inputs.txt", "dataset1_outputs.txt")

plt.plot(x, y, "x", color="green")
plt.show()

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
    identity = lam * np.identity(d)
    phi_t = np.dot(np.transpose(phi), phi)
    inner = np.add(identity, phi_t)
    outer = np.dot(np.linalg.inv(inner), np.transpose(phi))
    w = np.dot(outer, y)
    return w

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


for D in range(1, num_basis + 1):
    phi = generate_phi(D, x)
    w_mle = generate_w_mle(phi, y)
    w_map = generate_w_map(phi, y, D)
    pred_y_mle = np.zeros(len(y))
    pred_y_map = np.zeros(len(y))
    for i in range(len(y)):
        pred_y_mle[i] = compute_polynomials(w_mle, x[i], D)
        pred_y_map[i] = compute_polynomials(w_map, x[i], D)

    data_mle = sorted(itertools.izip(*[x, pred_y_mle]))
    mle_x, mle_y = list(itertools.izip(*data_mle))
    x_smooth = np.linspace(min(mle_x), max(mle_x), 200)
    y_smooth = spline(mle_x, mle_y, x_smooth)
    plt.plot(x_smooth, y_smooth, "-", label="MLE", color="red")

    data_map = sorted(itertools.izip(*[x, pred_y_map]))
    map_x, map_y = list(itertools.izip(*data_map))
    x_smooth = np.linspace(min(map_x), max(map_x), 200)
    y_smooth = spline(map_x, map_y, x_smooth)
    plt.plot(x_smooth, y_smooth, "-", label="MAP", color="blue")

    plt.plot(x, y, "x", label="Data", color="green")
    plt.title("D = {}".format(D))
    plt.legend()
    plt.show()


# Step 5

fold = 10
avg = np.zeros(num_basis)

for i in range(1, num_basis + 1):
    mses = np.zeros(10)
    for j in range(0, len(x), fold):

        training = np.concatenate([x[0:j], x[j+fold : len(x)]])
        training_y = np.concatenate([y[0:j], y[j+fold: len(y)]])

        validation = x[j:j + fold]
        valid_y = y[j:j + fold]

        phi = generate_phi(i, training)
        w_map = generate_w_map(phi, training_y, i)

        mses[j/fold] = compute_mse(i, w_map, valid_y, validation)

    avg[i-1] = np.mean(mses)

print np.argmin(avg)
plt.plot(range(0, 20), avg)
plt.title("Ten Fold Cross Validation Average MSE results")
plt.xlabel("D")
plt.ylabel("MSE")
plt.show()

# Step 6

x, y = load_data("dataset2_inputs.txt", "dataset2_outputs.txt")
plt.plot(x, y, "x", color="green")
plt.show()


# Step 7

tau = 1000
sigma = 1.5
d = 13

phi = generate_phi(d, x)
ide = tau**(-2) * np.identity(d)
sig = sigma**(-2) * np.dot(np.transpose(phi), phi)
inv = np.add(ide, sig)
cov_w = np.linalg.inv(inv)
mu_w = sigma**(-2) * np.dot(np.dot(cov_w, np.transpose(phi)), y)

mu_d = np.zeros(len(x))
sig_d = np.zeros(len(x))
plus_sig = np.zeros(len(x))
minus_sig = np.zeros(len(x))

for i in range(len(x)):
    mu_d[i] = np.dot(np.transpose(mu_w), phi[i])
    sig_d[i] = sigma**2 + np.dot(np.dot(np.transpose(phi[i]), cov_w), phi[i])
    plus_sig[i] = mu_d[i] + sig_d[i]
    minus_sig[i] = mu_d[i] - sig_d[i]

plt.plot(x, y, "x", color="green")

data_mu = sorted(itertools.izip(*[x, mu_d]))
mu_x, mu_y = list(itertools.izip(*data_mu))
x_smooth = np.linspace(min(mu_x), max(mu_x), 200)
y_smooth = spline(mu_x, mu_y, x_smooth)
plt.plot(x_smooth, y_smooth, "-", color="blue")

data_plus = sorted(itertools.izip(*[x, plus_sig]))
plus_x, plus_y = list(itertools.izip(*data_plus))
x_smooth = np.linspace(min(plus_x), max(plus_x), 200)
y_smooth = spline(plus_x, plus_y, x_smooth)
plt.plot(x_smooth, y_smooth, "--", color="green")

data_minus = sorted(itertools.izip(*[x, minus_sig]))
minus_x, minus_y = list(itertools.izip(*data_minus))
x_smooth = np.linspace(min(minus_x), max(minus_x), 200)
y_smooth = spline(minus_x, minus_y, x_smooth)
plt.plot(x_smooth, y_smooth, "--", color="green")

plt.show()