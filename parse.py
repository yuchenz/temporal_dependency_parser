import sys
import os
import operator
import codecs
from data_preparation import make_test_data
from ffNN_classifier import FFNN_Classifier
from data_structures import Action


def output_parse(edge_list, snt_list, output_file):
    with codecs.open(output_file, 'a', 'utf-8') as f:
        text = '\n'.join(snt_list)
        edge_text = '\n'.join(edge_list)
        f.write('SNT_LIST\n' + text + '\n' + 'EDGE_LIST\n' + edge_text + '\n\n')


def decode(test_data, classifier, output_file):
    #import pdb; pdb.set_trace()
    for state in test_data:
        edge_list = []
        while state.buffer_ or len(state.stack) > 1:
            action, value = max(enumerate(classifier.predict(state).npvalue()),
                key=operator.itemgetter(1))

            if action == Action.SHIFT:
                state.shift()

            elif action == Action.LEFT_REDUCE:
                edge = '\t'.join([state.stack[-1].ID, state.stack[-1].label,
                    state.stack[-2].ID, 'link'])
                edge_list.append(edge)

                state.left_reduce()

            elif action == Action.RIGHT_REDUCE:
                edge = '\t'.join([state.stack[-2].ID, state.stack[-2].label,
                    state.stack[-1].ID, 'link'])
                edge_list.append(edge)

                state.right_reduce()

            elif action == Action.ROOT_REDUCE:
                edge = '\t'.join([state.stack[-1].ID, state.stack[-1].label,
                    state.stack[0].ID, 'link'])
                edge_list.append(edge)

                state.root_reduce()

        edge_list = sorted(edge_list,
            key = lambda x: tuple([int(i) for i in x.split()[0].split('_')]))

        output_parse(edge_list, state.snt_list, output_file)


if __name__ == '__main__':
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    vocab_file = sys.argv[3]

    test_data = make_test_data(test_file)

    classifier = FFNN_Classifier.load_model(model_file, vocab_file)

    parsed_file = test_file + '.parsed'

    try:
        os.remove(parsed_file)
    except OSError:
        pass

    decode(test_data, classifier, parsed_file)
