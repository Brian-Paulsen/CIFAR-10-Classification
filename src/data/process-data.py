import os
import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

def create_master_dataset(directory):
    batchFormat = os.path.join(directory, 'data_batch_{}')

    dataLists = []
    for i in range(1, 6):
        data = unpickle(batchFormat.format(i))
        dataLists.append(data)

    dataPoints = sum(len(l[b'labels']) for l in dataLists)
    labels = np.zeros((dataPoints,))
    data = np.zeros((dataPoints,3072))
    
    nextEmpty = 0
    for d in dataLists:
        n = len(dataLists[0][b'labels'])
        labels[nextEmpty:(nextEmpty+n)] = d[b'labels']
        data[nextEmpty:(nextEmpty+n),] = d[b'data']

    return data, labels

# create_master_dataset('../../data/raw/cifar-10-batches-py')

