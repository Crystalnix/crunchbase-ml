import os


def path_to_data_file(path):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.normpath(os.path.join(cur_dir, path))

data = '../data/data.csv'
train = '../data/train_data.csv'
test = '../data/train_data.csv'


DATA_FILE = path_to_data_file(data)
TRAIN_FILE = path_to_data_file(train)
TEST_FILE = path_to_data_file(test)

