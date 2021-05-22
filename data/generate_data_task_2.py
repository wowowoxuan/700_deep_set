import random
import numpy as np

def generate_training_data(num_samples):
    train_label=[] 
    training_set = []
    for i in range(num_samples):
        print(i)
        length = random.randint(1,10)
        print(length)
        temp_list = []
        temp_label = []
        # set m=2 ==> repeat 2 times ==> 3 times in total
        for j in range(length):
            idx = random.randint(0,25)
            onehot = [0] * 26
            onehot[idx] = 1
            temp_list.append(onehot)
            temp_label.append(onehot)
            temp_label.append(onehot)
            # temp_label.append(onehot)
        print(temp_list)
        training_set.append(temp_list)
        # temp_label = temp_label * 2
        train_label.append(temp_label)
    np.save('./data/train.npy',np.array(training_set))
    np.save('./data/train_label.npy',np.array(train_label))
    return np.array(training_set)


def generate_val_data(max_len, num_samples):
    train_label=[] 
    training_set = []
    for i in range(num_samples):
        print(i)
        length = random.randint(1,max_len)
        temp_list = []
        temp_label = []
        # set m=2 ==> repeat 2 times ==> 3 times in total

        for j in range(length):
            idx = random.randint(0,25)
            onehot = [0] * 26
            onehot[idx] = 1
            temp_list.append(onehot)
            temp_label.append(onehot)
            temp_label.append(onehot)
            # temp_label.append(onehot)
        training_set.append(temp_list)
        # temp_label = temp_label * 2
        train_label.append(temp_label)
    np.save('./data/val.npy',np.array(training_set))
    np.save('./data/val_label.npy',np.array(train_label))
    return np.array(training_set)

def generate_testing_data(max_len, num_samples):
    train_label=[] 
    training_set = []
    for i in range(num_samples):
        print(i)
        length = random.randint(1,max_len)
        temp_list = []
        temp_label = []
        # set m=2 ==> repeat 2 times ==> 3 times in total

        for j in range(length):
            idx = random.randint(0,25)
            onehot = [0] * 26
            onehot[idx] = 1
            temp_list.append(onehot)
            temp_label.append(onehot)
            # temp_label.append(onehot)
            temp_label.append(onehot)
        training_set.append(temp_list)
        # temp_label = temp_label * 2
        train_label.append(temp_label)
    np.save('./data/test.npy',np.array(training_set))
    np.save('./data/test_label.npy',np.array(train_label))
    return np.array(training_set)
# train = generate_training_data(10000)
# test = generate_testing_data(100,50000)
train = generate_training_data(10000)
val = generate_val_data(10,5000)
test = generate_testing_data(25, 5000)



# print(test.shape)

# np_load_old = np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# a = np.load('./val.npy')
# # print(a==b)
# a = a.tolist()
# print(len(a))