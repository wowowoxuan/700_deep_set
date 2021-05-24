import random
import numpy as np
import os

if not os.path.exists('./task_3_data'):
    os.mkdir('./task_3_data')


def generate_training_data(num_samples):
    training_set = []
    training_label = []
    for i in range(num_samples):
        #print('sample: ',i)
        length = random.randint(1, 10)
        #print('len: ',length)
        temp_list = []

        for j in range(length):
            idx = random.randint(0, 25)
            onehot = [0] * 26
            onehot[idx] = 1
            temp_list.append(onehot)
        temp_label=temp_list.copy()
        temp_label.reverse()
        #print('reverse',temp_label)
        #temp_list.append(endChar)
        #print('data',temp_list)
        #print('label',temp_label)
        training_set.append(temp_list)
        training_label.append(temp_label)
    np.save('./task_3_data/task_train.npy', np.array(training_set))
    np.save('./task_3_data/train_label.npy', np.array(training_label))
    print('Generating training data success!')
    return np.array(training_set)


def generate_val_data(max_len, num_samples):
    training_set = []
    training_label = []
    for i in range(num_samples):
        # print('sample: ',i)
        length = random.randint(1, max_len)
        # print('len: ',length)
        temp_list = []

        for j in range(length):
            idx = random.randint(0, 25)
            onehot = [0] * 26
            onehot[idx] = 1
            temp_list.append(onehot)
        temp_label = temp_list.copy()
        temp_label.reverse()
        # print('reverse',temp_label)
        #temp_list.append(endChar)
        # print('data',temp_list)
        # print('label',temp_label)
        training_set.append(temp_list)
        training_label.append(temp_label)
    np.save('./task_3_data/val.npy', np.array(training_set))
    np.save('./task_3_data/val_label.npy', np.array(training_label))
    print('Generating validation data success!')
    return np.array(training_set)


def generate_test_data(max_len, num_samples):
    training_set = []
    training_label = []
    for i in range(num_samples):
        # print('sample: ',i)
        length = random.randint(1, max_len)
        # print('len: ',length)
        temp_list = []

        for j in range(length):
            idx = random.randint(0, 25)
            onehot = [0] * 26
            onehot[idx] = 1
            temp_list.append(onehot)
        temp_label = temp_list.copy()
        temp_label.reverse()
        # print('reverse',temp_label)
        # temp_list.append(endChar)
        # print('data',temp_list)
        # print('label',temp_label)
        training_set.append(temp_list)
        training_label.append(temp_label)
    np.save('./task_3_data/test.npy', np.array(training_set))
    np.save('./task_3_data/test_label.npy', np.array(training_label))
    print('Generating testing data success!')
    return np.array(training_set)


def generate_testing_fix_data(length, num_samples):
    training_set = []
    training_label = []
    for i in range(num_samples):
        # print('sample: ',i)
        # print('len: ',length)
        temp_list = []

        for j in range(length):
            idx = random.randint(0, 25)
            onehot = [0] * 26
            onehot[idx] = 1
            temp_list.append(onehot)
        temp_label = temp_list.copy()
        temp_label.reverse()
        # print('reverse',temp_label)
        # temp_list.append(endChar)
        # print('data',temp_list)
        # print('label',temp_label)
        training_set.append(temp_list)
        training_label.append(temp_label)
    np.save('./task_3_data/test_' + str(length) + '.npy', np.array(training_set))
    np.save('./task_3_data/test_label_' + str(length) + '.npy', np.array(training_label))
    print('Generating test_fixing data success!')
    return np.array(training_set)


# train = generate_training_data(10000)
# test = generate_testing_data(100,50000)

#train = generate_training_data(5000)
train = generate_training_data(5000)
val = generate_val_data(10,5000)
test = generate_test_data(90,50000)
for i in range(1,15):
    generate_testing_fix_data(i*10,5000)
# np_load_old = np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# a = np.load('./val.npy')
# # print(a==b)
# a = a.tolist()
# print(len(a))