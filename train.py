import sys
from data_preparation import make_training_data
from ffNN_classifier import FFNN_Classifier


if __name__ == '__main__':
    train_file = sys.argv[1]

    training_data, vocab = make_training_data(train_file)

    for doc in training_data:
        for state, action in doc:
            print(state)
            print(action, end='\n\n')

    classifier = FFNN_Classifier(8, 8, 4, vocab)
    classifier.train(training_data, 'ffnn_model', num_iter=10)
