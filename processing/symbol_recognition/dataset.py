import numpy as np
import pandas as pd


# датасет изначально повернут и отражен
def rotate(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image


def del_not_useful(x, y):
    values_y = np.apply_along_axis(np.argmax, 1, y)

    A_J_K_Q_DIG = {10, 19, 20, 26}
    DIGITS = {i for i in range(10)}
    A_J_K_Q_DIG.update(DIGITS)

    bool_y = np.isin(values_y, list(A_J_K_Q_DIG))
    useful_indexes = np.where(bool_y)

    values_y = values_y[useful_indexes]
    values_y[values_y == 19] = 11
    values_y[values_y == 20] = 12
    values_y[values_y == 26] = 13
    values_y = pd.get_dummies(values_y)

    return x[useful_indexes], values_y


def get_dataset():
    print('reading data...')
    train = pd.read_csv('dataset/emnist-balanced-train.csv', header=None)
    test = pd.read_csv('dataset/emnist-balanced-test.csv', header=None)

    print('trasform...')
    train_data = train.iloc[:, 1:]
    train_labels = train.iloc[:, 0]
    test_data = test.iloc[:, 1:]
    test_labels = test.iloc[:, 0]

    train_labels = pd.get_dummies(train_labels)
    test_labels = pd.get_dummies(test_labels)

    train_data = train_data.values
    train_labels = train_labels.values
    test_data = test_data.values
    test_labels = test_labels.values
    del train, test

    print('rotating...')
    train_data = np.apply_along_axis(rotate, 1, train_data)
    test_data = np.apply_along_axis(rotate, 1, test_data)

    print('deleting unuseful...')
    train_data, train_labels = del_not_useful(train_data, train_labels)
    test_data, test_labels = del_not_useful(test_data, test_labels)

    return (train_data, train_labels), (test_data, test_labels)

