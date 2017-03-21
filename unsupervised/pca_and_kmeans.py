import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# STEP 1

matfile = loadmat('face_images.mat')

train_imgs = matfile['train_imgs']
test_imgs = matfile['test_imgs']
train_ids = matfile['train_ids']
test_ids = matfile['test_ids']

# for i in range(0, 3):
#     plt.imshow(train_imgs[i])
#     plt.show()

b = np.mean(train_imgs)
# plt.imshow(b)
# plt.show()

X = 1.0/np.sqrt(len(train_imgs)) * (train_imgs - b)

eigenvectors, eigenvalues, variance = np.linalg.svd(X, full_matrices=False)
eigs, vecs = np.linalg.eig(np.dot(eigenvalues, eigenvalues.T))

# svd = np.dot(X.T, X)
# [eigvals, eigvecs] = np.linalg.eigh(svd)
# eigs, vecs = np.linalg.eig(np.dot(s, np.transpose(s)))
# eigs, vecs = np.linalg.eig(s)

sorted_eigs = sorted(eigs, reverse=True)

# idx = np.argsort(-eigs)
# eigenvalues = sorted(eigs, reverse=True)

plt.plot(range(len(sorted_eigs)), sorted_eigs)
plt.show()


def fraction_of_variances(eigs):
    sum_eigs = np.sum(eigs)
    dim_subspace = None
    fracs = np.zeros(len(eigs))
    for i in range(0, len(eigs)):
        fracs[i] = np.sum(eigs[:i+1])/sum_eigs
        if dim_subspace is None and fracs[i] >= 0.9:
            dim_subspace = i
    return fracs, dim_subspace

frac_of_vars, dimensionality = fraction_of_variances(sorted_eigs)
plt.plot(range(len(frac_of_vars)), frac_of_vars)
plt.show()


for i in range(10):
    # W = np.diag(eigs)
    x = np.dot(sorted_eigs[i].T, (train_imgs[0] - b))
    # plt.imshow(x)
    # plt.show()


# STEP 3

x = np.dot(sorted_eigs[:dimensionality], (train_imgs - b))

y_recon = b + np.dot(sorted_eigs[:dimensionality], x)

error = abs(y_recon - train_imgs)**2

sample = np.random.permutation(range(len(train_imgs)))

for i in range(5):
    index = sample[i]

    plt.title("Square Reconstruction Error: " + error[index])
    plt.subplot(211)
    plt.imshow(train_imgs[index])
    plt.subplot(212)
    plt.imshow(y_recon[index])
    plt.subplot(213)
    abs_dif = np.abs(train_imgs[index] - y_recon[index])
    plt.imshow(abs_dif[index])
    plt.show()

plt.hist(error, bins="auto")
plt.show()




