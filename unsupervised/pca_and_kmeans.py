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


input_temp = []
for i in range(len(train_imgs)):
    input_temp.append(train_imgs[i].reshape(1, -1)[0])

input_train = np.array(input_temp)


def generate_b(image_set):
    b = np.zeros(200*180)
    for i in range(len(image_set)):
        b += image_set[i]
    b /= len(image_set)
    return b

b = generate_b(input_train)

# plt.imshow(b.reshape((200, 180))
# plt.show()

X =  1.0/np.sqrt(len(input_train))*(input_train - b)

eigvecs, eigs, var = np.linalg.svd(X.T, full_matrices=False)


sorted_eigs = sorted(eigs, reverse=True)

# plt.plot(range(len(sorted_eigs)), sorted_eigs)
# plt.show()


def fraction_of_variances(eigs):
    sum_eigs = np.sum(eigs)
    cum_sum = np.cumsum(eigs)/sum_eigs
    dim_subspace = np.where(cum_sum >= 0.9)[0][0]
    return cum_sum, dim_subspace

frac_of_vars, dimensionality = fraction_of_variances(sorted_eigs)
# plt.plot(range(len(frac_of_vars)), frac_of_vars)
# plt.show()


# for i in range(10):

    # plt.imshow(eigvecs[:, i].reshape((200, 180)))
    # plt.show()


# STEP 3

W = eigvecs[:, 0: dimensionality]
x = np.dot(W.T, (input_train - b).T)

y_recon = b + np.dot(x.T, W.T)

error = abs(y_recon - input_train)**2

sample = np.random.permutation(range(len(train_imgs)))

for i in range(5):
    index = sample[i]

    plt.title("Square Reconstruction Error: {}".format(sum(error[index])))
    plt.subplot(131)
    plt.imshow(input_train[index].reshape((200, 180)))
    plt.subplot(132)
    plt.imshow(y_recon[index].reshape((200, 180)))
    plt.subplot(133)
    abs_dif = abs(y_recon[index] - input_train[index])
    plt.imshow(abs_dif.reshape((200, 180)))
    plt.show()

# plt.hist(error, bins="auto")
# plt.show()


# matfile_non = loadmat('nonface_images.mat')
#
# train_imgs_non = matfile['train_imgs']
# test_imgs_non = matfile['test_imgs']
# train_ids_non = matfile['train_ids']
# test_ids_non = matfile['test_ids']
#
# x_non = np.dot(W, (train_imgs_non - b))
# y_recon = b + np.dot(W, x_non)
#
# error = abs(y_recon - train_imgs)**2
#
# sample = np.random.permutation(range(len(train_imgs)))
#
# for i in range(5):
#     index = sample[i]
#
#     plt.title("Square Reconstruction Error: " + error[index])
#     plt.subplot(211)
#     plt.imshow(train_imgs[index])
#     plt.subplot(212)
#     plt.imshow(y_recon[index])
#     plt.subplot(213)
#     abs_dif = np.abs(train_imgs[index] - y_recon[index])
#     plt.imshow(abs_dif[index])
#     plt.show()
#
# plt.hist(error, bins="auto")
# plt.show()

