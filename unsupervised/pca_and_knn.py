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


def reshape_image_matrix(train_imgs):
    input_temp = []
    for i in range(len(train_imgs)):
        input_temp.append(train_imgs[i].reshape(1, -1)[0])

    return np.array(input_temp)


input_train = reshape_image_matrix(train_imgs)


def generate_b(image_set):
    b = np.zeros(200*180)
    for i in range(len(image_set)):
        b += image_set[i]
    b /= len(image_set)
    return b


def pca(image_set):
    b = generate_b(image_set)
    X = image_set - b
    eigvecs, eigs, var = np.linalg.svd(X.T, full_matrices=False)
    idx = eigs.argsort()[::-1]
    eigs = eigs[idx]
    eigvecs = eigvecs[:, idx]
    sorted_eigs = sorted(eigs, reverse=True)

    return b, eigvecs, sorted_eigs

b, eigvecs, eigs = pca(input_train)

# plt.imshow(b.reshape((200, 180)))
# plt.show()

# plt.plot(range(len(eigs)), eigs)
# plt.show()


def fraction_of_variances(eigs):
    sum_eigs = np.sum(eigs)
    cum_sum = np.cumsum(eigs)/sum_eigs
    dim_subspace = np.where(cum_sum >= 0.9)[0][0]
    return cum_sum, dim_subspace

frac_of_vars, dimensionality = fraction_of_variances(eigs)
# plt.plot(range(len(frac_of_vars)), frac_of_vars)
# plt.show()

dimensionality = 115

# for i in range(10):
    # plt.imshow(eigvecs[:, i].reshape((200, 180)))
    # plt.show()


# STEP 3

W = eigvecs[:, 0: dimensionality]


def reconstruct_from_basis(W, b, image_set):

    x = np.dot(W.T, (image_set - b).T)
    y_recon = b + np.dot(x.T, W.T)
    error = np.power(abs(y_recon - image_set), 2)
    sample = np.random.permutation(range(len(image_set)))
    for i in range(5):
        index = sample[i]

        plt.suptitle("Square Reconstruction Error: {}".format(np.sum(error[index])))
        plt.subplot(131)
        plt.imshow(image_set[index].reshape((200, 180)))
        plt.subplot(132)
        plt.imshow(y_recon[index].reshape((200, 180)))
        plt.subplot(133)
        abs_dif = abs(y_recon[index] - image_set[index])
        plt.imshow(abs_dif.reshape((200, 180)))
        plt.show()
    return y_recon, error

y__face_recon, face_error = reconstruct_from_basis(W, b, input_train)

face_error_sums = [np.sum(face_error[i]) for i in range(len(face_error))]

# Load non face images
matfile_non = loadmat('nonface_images.mat')

train_imgs_non = matfile_non['nonface_imgs']

non_input_train = reshape_image_matrix(train_imgs_non)

y_non_recon, non_error = reconstruct_from_basis(W, b, non_input_train)

non_error_sums = [sum(non_error[i]) for i in range(len(non_error))]

# Show errors in histogram
# bins =
plt.hist(non_error_sums, bins='auto',  label='x')
plt.hist(face_error_sums, bins='auto',  label='y')
plt.legend(loc='upper right')
plt.show()

# Classify face images as non face, and vice versa
threshold = 7370
