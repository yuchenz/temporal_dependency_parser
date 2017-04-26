import random
import codecs
import json
from data_structures import Action
from vector import Vector


class Perceptron_Classifier:
    """ Perceptron Classifier. """

    def __init__(self, vocab):
        self.weights = Vector({})
        self.vocab = vocab

    @classmethod
    def load_model(cls, model_file, vocab_file):
        classifier = cls(None)
        classifier.weights = Vector({})
        classifier.weights.load(model_file)

        with codecs.open(vocab_file, 'r', 'utf-8') as f:
            classifier.vocab = json.load(f) 

        return classifier

    def train(self, training_data, output_file, num_iter=1000):
        for i in range(num_iter):
            random.shuffle(training_data)
            closs = 0.0
            for example in training_data:
                for state, action in example:
                    yhat = self.predict(state)
                    good = self.extract_feature_vec(state, action)
                    self.weights += good 
                    bad = self.extract_feature_vec(state, yhat)
                    self.weights -= bad 
                    #print('good ', good)
                    #print('bad ', bad)
        
        self.weights.save(output_file)
        with codecs.open(output_file + '.vocab', 'w', 'utf-8') as f:
            json.dump(self.vocab, f)

    def predict(self, state):
        yhat = []
        for action in [Action.SHIFT, Action.LEFT_REDUCE,
                Action.RIGHT_REDUCE, Action.ROOT_REDUCE]:
            x = self.extract_feature_vec(state, action)
            score = self.weights.dot(x)
            yhat.append(score)

        return max(enumerate(yhat), key=lambda x: x[1])[0]
    
    def extract_feature_vec(self, state, action):
        vec = Vector({})
        action = str(action)

        words = state.stack[-2].words if len(state.stack) > 1 else '<START>'
        feat = words if words in self.vocab else '<UNK>'
        vec.v['s2=' + feat + '+a=' + action] = 1

        words = state.stack[-1].words if len(state.stack) > 0 else '<START>'
        feat = words if words in self.vocab else '<UNK>'
        vec.v['s1=' + feat + '+a=' + action] = 1

        words = state.buffer_[0].words if len(state.buffer_) > 0 else '<STOP>'
        feat = words if words in self.vocab else '<UNK>'
        vec.v['b1=' + feat + '+a=' + action] = 1
        
        vec.v['a=' + action] = 1
        #print('=================vec:', vec)

        return vec 
