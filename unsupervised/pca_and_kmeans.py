import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# STEP 1

matfile = loadmat('face_images.mat')

train_imgs = matfile['train_imgs']
test_imgs = matfile['test_imgs']
train_ids = matfile['train_ids']
test_ids = matfile['test_ids']

for i in range(0, 3):
    plt.imshow(train_imgs[i])
    plt.show()


def find_average_image(image_set):
    avg = np.zeros((200, 180))

    for i in image_set:
        avg += i

    avg /= len(image_set)
    return avg

avg = find_average_image(train_imgs)
plt.imshow(avg)
plt.show()


