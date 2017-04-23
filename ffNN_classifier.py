import dynet as dy
import random


class FFNN_Classifier:
    """ Feed-Forward Neural Network Classifier. """

    def __init__(self, embed_size, hidden_size, output_size, vocab):
        self.model = dy.Model()

        self.embeddings = self.model.add_lookup_parameters((len(vocab), embed_size))

        self.pW1 = self.model.add_parameters((hidden_size, embed_size * 3))
        self.pb = self.model.add_parameters(hidden_size)
        self.pW2 = self.model.add_parameters((output_size, hidden_size))

        self.vocab = vocab


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
        
        self.model.save(output_file, [self.embeddings, self.pW1, self.pb, self.pW2]) 


    def predict(self, state):
        x = self.extract_feature_vec(state)

        W1 = dy.parameter(self.pW1)
        b = dy.parameter(self.pb)
        W2 = dy.parameter(self.pW2)
        
        hidden = dy.tanh(W1 * x + b)
        yhat = W2 * hidden

        return yhat


    def compute_loss(self, yhat, action):
        return -dy.log(dy.pick(dy.softmax(yhat), action))

    
    def extract_feature_vec(self, state):
        feat = self.vocab.get(state.stack[-2].words, self.vocab['<UNK>']) \
            if len(state.stack) > 1 else self.vocab['<START>']
        x1 = dy.lookup(self.embeddings, feat)

        feat = self.vocab.get(state.stack[-1].words, self.vocab['<UNK>']) \
            if len(state.stack) > 0 else self.vocab['<START>']
        x2 = dy.lookup(self.embeddings, feat)

        feat = self.vocab.get(state.buffer_[0].words, self.vocab['<UNK>']) \
            if len(state.buffer_) > 0 else self.vocab['<STOP>']
        x3 = dy.lookup(self.embeddings, feat)

        feat_vec = dy.concatenate([x1, x2, x3])

        return feat_vec
