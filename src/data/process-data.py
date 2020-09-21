import pickle

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

batch1 = '../../data/raw/cifar-10-batches-py/data_batch_1'
data = unpickle(batch1)

