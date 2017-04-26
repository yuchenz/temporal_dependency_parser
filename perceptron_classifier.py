import dynet as dy
import random
import codecs
import json
from data_structures import Action


class Perceptron_Classifier:
    """ Perceptron Classifier. """

    def __init__(self, vocab):
        self.model = dy.Model()

        if vocab != None:
            self.vocab = vocab
            self.vocab_size = len(vocab)
            self.p_theta = self.model.add_parameters(self.vocab_size * 3 + 4) 
            self.pb = self.model.add_parameters(1)  # bias
        else:
            self.p_theta, self.pb, self.vocab, self.vocab_size = None, None, None, 0

    @classmethod
    def load_model(cls, model_file, vocab_file):
        classifier = cls(None)
        classifier.p_theta, classifier.pb = classifier.model.load(model_file)

        with codecs.open(vocab_file, 'r', 'utf-8') as f:
            classifier.vocab = json.load(f) 
            classifier.vocab_size = len(classifier.vocab)

        return classifier


    def train(self, training_data, output_file, num_iter=1000):
        trainer = dy.SimpleSGDTrainer(self.model)

        for i in range(num_iter):
            random.shuffle(training_data)
            closs = 0.0
            for example in training_data:
                for state, action in example:
                    dy.renew_cg()

                    yhat = self.predict(state)

                    loss = self.compute_loss(yhat, action)

                    closs += loss.scalar_value()
                    loss.backward()
                    trainer.update()
        
        self.model.save(output_file, [self.p_theta, self.pb]) 
        with codecs.open(output_file + '.vocab', 'w', 'utf-8') as f:
            json.dump(self.vocab, f)


    def predict(self, state):
        theta = dy.parameter(self.p_theta)
        b = dy.parameter(self.pb)

        yhat = []
        for action in [Action.SHIFT, Action.LEFT_REDUCE,
                Action.RIGHT_REDUCE, Action.ROOT_REDUCE]:
            x = self.extract_feature_vec(state, action)
            yhat.append(dy.dot_product(theta, x) + b)

        return dy.concatenate(yhat)


    def compute_loss(self, yhat, action):
        return -dy.pick(yhat, action) + max(yhat.npvalue())

    
    def extract_feature_vec(self, state, action):
        feat = self.vocab.get(state.stack[-2].words, self.vocab['<UNK>']) \
            if len(state.stack) > 1 else self.vocab['<START>']
        feat_vec = [1 if i == feat else 0 for i in range(self.vocab_size)]
        x1 = dy.inputVector(feat_vec)

        feat = self.vocab.get(state.stack[-1].words, self.vocab['<UNK>']) \
            if len(state.stack) > 0 else self.vocab['<START>']
        feat_vec = [1 if i == feat else 0 for i in range(self.vocab_size)]
        x2 = dy.inputVector(feat_vec)

        feat = self.vocab.get(state.buffer_[0].words, self.vocab['<UNK>']) \
            if len(state.buffer_) > 0 else self.vocab['<STOP>']
        feat_vec = [1 if i == feat else 0 for i in range(self.vocab_size)]
        x3 = dy.inputVector(feat_vec)

        feat_vec = [1 if i == action else 0 for i in range(4)]
        x4 = dy.inputVector(feat_vec)

        feat_vec = dy.concatenate([x1, x2, x3, x4])

        return feat_vec
