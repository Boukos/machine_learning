import numpy as np
import csv
from collections import OrderedDict
import matplotlib.pyplot as plt


filename = 'mushrooms.csv'
with open(filename, 'rb') as raw_file:
    raw_data = csv.reader(raw_file, delimiter=',', quoting=csv.QUOTE_NONE)
    data_list = list(raw_data)

ndims = len(data_list[0])
npts = len(data_list)

char_maps = [OrderedDict() for i in range(ndims)] # Mapping of character to integer
reverse_maps = [[] for i in range(ndims)] # Array of all the possible values for each field
data_mat = np.empty((npts,ndims), dtype=np.int32) # Converted matrix of csv
for i,cdata in enumerate(data_list):
    for j,cstr in enumerate(cdata):
        if cstr not in char_maps[j]:
            char_maps[j][cstr] = len(char_maps[j])
            reverse_maps[j].append(cstr)
        data_mat[i,j] = char_maps[j][cstr]
del data_list

np.random.seed(0)
data_perm = np.random.permutation(npts)
data_train = data_mat[data_perm[0:(8*npts/10)],:]
data_test = data_mat[data_perm[(8*npts/10):],:]
data_ranges = data_mat[:,1:].max(axis=0)


# STEP 1

def generate_poison_mat(data_train, data_mat):
    poison_array = np.where(data_train[:, 0] == 0)[0]
    len_psn = len(poison_array)
    poison_mat = np.empty((len_psn,len(data_mat[0])), dtype=np.int32)

    j = 0
    for i in poison_array:
        poison_mat[j] = data_train[i]
        j += 1

    return poison_mat


def generate_edible_mat(data_train, data_mat):
    edible_array = np.where(data_train[:, 0] == 1)[0]
    len_ed = len(edible_array)
    edible_mat = np.empty((len_ed, len(data_mat[0])), dtype=np.int32)

    j = 0
    for i in edible_array:
        edible_mat[j] = data_train[i]
        j += 1

    return edible_mat

features = ["cap-shape", "cap-surface", "cap-color", "bruises?", "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color",
            "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring",
            "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]

poison_mat = generate_poison_mat(data_train, data_mat)    # Matrix of poison training data
edible_mat = generate_edible_mat(data_train, data_mat)    # Matrix of edible training data

size_train = len(data_train)                                # N of all training data

poison_perc = float(len(poison_mat))/size_train             # Percentage of training data that is poisonous
edible_perc = float(len(edible_mat))/size_train             # Percentage of training data that is edible
print "Out of {}, {}% are poisonous and {}% are edible".format(size_train, poison_perc*100, edible_perc*100)


def show_histograms():
    for i in range(1, len(features) + 1):
        num_bins = len(char_maps[i])
        ed_hist = np.zeros(num_bins)
        po_hist = np.zeros(num_bins)

        for j in range(num_bins):
            ed_hist[j] = len(np.where(edible_mat[:, i] == j)[0])
            po_hist[j] = len(np.where(poison_mat[:, i] == j)[0])

        plt.subplot(211)
        plt.ylabel("Number of Occurences")
        plt.title("Feature {}: {}. poisonous".format(i, features[i-1]))
        plt.bar(range(num_bins), po_hist)

        plt.subplot(212)
        plt.ylabel("Number of Occurences")
        plt.title("Feature {}: {}. edible".format(i, features[i-1]))
        plt.bar(range(num_bins), ed_hist)
        plt.show()

show_histograms()

# STEP 2


def calculate_likelihoods(a):
    poisonous_features = [[] for i in range(ndims - 1)]  # Matrix of posterior values for poisonous features
    edible_features = [[] for i in range(ndims - 1)]  # Matrix of posterior values for edible features
    pN = len(poison_mat)  # Amount of poisonous values
    eN = len(edible_mat)  # Amount of edible values

    for i in range(1, len(features) + 1):

        num_bins = len(char_maps[i])

        ptheta = np.zeros(num_bins)
        etheta = np.zeros(num_bins)

        for j in range(num_bins):
            ed_amount = len(np.where(edible_mat[:, i] == j)[0])
            po_amount = len(np.where(poison_mat[:, i] == j)[0])

            etheta[j] = (float(ed_amount) + a - 1)/(eN + num_bins * a - num_bins)
            ptheta[j] = (float(po_amount) + a - 1)/(pN + num_bins * a - num_bins)

        poisonous_features[i - 1] = ptheta
        edible_features[i - 1] = etheta
    return poisonous_features, edible_features


def log_sum_exp(poisonous_features, edible_features, data):

    amount_correct = 0

    for i in range(len(data)):
        lp1 = 0
        lp0 = 0
        for j in range(1, len(data[0])):
            feature_val = data[i][j]

            if edible_features[j - 1][feature_val] != 0:
                lp1 += np.log(edible_features[j - 1][feature_val])

            if poisonous_features[j-1][feature_val] != 0:
                lp0 += np.log(poisonous_features[j-1][feature_val])

        lp1 += np.log(edible_perc)
        lp0 += np.log(poison_perc)
        B = max(lp1, lp0)
        posterior = lp1 - (np.log(np.exp(lp1 - B) + np.exp(lp0 - B)) + B)

        if posterior > np.log(0.5):
            if data[i][0] == 1:
                amount_correct += 1
        if posterior <= np.log(0.5):
            if data[i][0] == 0:
                amount_correct += 1

    # print "Amount correct classifications: {}. Percentage correct is: {}".format(amount_correct, (float(amount_correct)/size_train)*100)
    return amount_correct


alpha = np.arange(1, 2.001, 0.001)


def best_alpha(alpha, data):
    best = -1000000
    best_a = -1000000
    all_alpha = np.zeros(len(alpha))
    size_data = len(data)
    i = 0
    for a in alpha:
        poisonous_features, edible_features = calculate_likelihoods(a)
        amount_accuracy = float(log_sum_exp(poisonous_features, edible_features, data))/size_data
        all_alpha[i] = amount_accuracy
        i += 1
        if amount_accuracy > best:
            best = amount_accuracy
            best_a = a

    return best_a, all_alpha

best_a_train, all_alpha_train = best_alpha(alpha, data_train)

best_a_test, all_alpha_test = best_alpha(alpha, data_test)


plt.xlabel("Alpha values")
plt.ylabel("Probability")
plt.ylim((0.87, 1.00))
plt.xlim((1.0, 2.0))
plt.plot(alpha, all_alpha_train, label="Train Set")
plt.plot(alpha, all_alpha_test, label="Validation Set")
plt.legend()
plt.show()


# STEP 3


def biggest_impact(a):
    computed_matrix = []            # Stores lists with structure [i, f_i, abs_value, raw_value]
    poisonous_features, edible_features = calculate_likelihoods(a)
    for i in range(len(poisonous_features)):
        for j in range(len(poisonous_features[i])):

            edible = 0
            poison = 0

            if np.log(edible_features[i][j]) != 0:
                edible = np.log(edible_features[i][j])

            if np.log(poisonous_features[i][j]) != 0:
                poison = np.log(poisonous_features[i][j])

            value = edible - poison
            abs_value = abs(value)
            computed_matrix.append([i + 1, j, abs_value, value])

    def sort_by_abs(item):
        return item[2]

    def sort_by_val(item):
        return item[3]

    abs_sorted = sorted(computed_matrix, key=sort_by_abs)
    raw_sorted = sorted(computed_matrix, key=sort_by_val)

    return abs_sorted, raw_sorted


abs_sorted, raw_sorted = biggest_impact(best_a_train)

print "Sorted absolute difference: {}".format(abs_sorted)

print "Sorted differences: {}".format(raw_sorted)





