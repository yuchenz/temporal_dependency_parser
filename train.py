import sys
import codecs
from data_preparation import make_training_data


if __name__ == '__main__':
    train_file = sys.argv[1]

    training_data = make_training_data(train_file)
    for doc in training_data:
        for state, action in doc:
            print(state)
            print(action, end='\n\n')
