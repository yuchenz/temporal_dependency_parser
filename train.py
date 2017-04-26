import sys
from data_preparation import make_training_data
from ffNN_classifier import FFNN_Classifier
from perceptron_classifier import Perceptron_Classifier


if __name__ == '__main__':
    train_file = sys.argv[1]

    training_data, vocab = make_training_data(train_file)

    #for doc in training_data:
    #    for state, action in doc:
    #        print(state)
    #        print(action, end='\n\n')

    clas = sys.argv[2]

    if clas == 'ffnn':
        classifier = FFNN_Classifier(8, 8, 4, vocab)
        classifier.train(training_data, 'ffnn_model', num_iter=10)
    elif clas == 'perceptron':
        classifier = Perceptron_Classifier(vocab)
        classifier.train(training_data, 'perceptron_model', num_iter=10)
