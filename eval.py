import codecs
import sys


def readin_tuples(filename):
    lines = codecs.open(filename, 'r', 'utf-8').readlines()

    edge_tuples = []
    mode = 0
    for line in lines:
        if line.startswith('SNT_LIST'):
            mode = 0
            edge_tuples.append([])
        elif line.startswith('EDGE_LIST'):
            mode = 1
        elif line.strip() == '':
            continue
        else:
            if mode:
                child, child_label, parent, link_label = line.strip().split()
                edge_tuples[-1].append((child, parent, link_label))

    return edge_tuples


def unlabelled_eval(gold_tuples, auto_tuples):
    counts = []
    scores = []

    for i, (gtups, atups) in enumerate(zip(gold_tuples, auto_tuples)):
        gtups = set([(gtup[0], gtup[1]) for gtup in gtups])
        atups = set([(atup[0], atup[1]) for atup in atups])

        true_positive = len(gtups.intersection(atups))
        false_positive = len(atups.difference(gtups))
        false_negative = len(gtups.difference(atups))

        print('test doc {}: true_p = {}, false_p = {}, false_n = {}'.format(
            i, true_positive, false_positive, false_negative)) 
        p = true_positive / (true_positive + false_positive)
        r = true_positive / (true_positive + false_negative)
        f = 2 * p * r / (p + r)

        counts.append((true_positive, false_positive, false_negative))
        scores.append((p, r, f))

    # macro average
    p = sum([score[0] for score in scores]) / len(scores)
    r = sum([score[1] for score in scores]) / len(scores)
    f = sum([score[2] for score in scores]) / len(scores)

    print('macro average: p = {}, r = {}, f = {}'.format(p, r, f))

    # micro average
    true_p = sum([count[0] for count in counts])
    false_p = sum([count[1] for count in counts])
    false_n = sum([count[2] for count in counts])

    p = true_p / (true_p + false_p)
    r = true_p / (true_p + false_n)
    f = 2 * p * r / (p + r)

    print('micro average: p = {}, r = {}, f = {}'.format(p, r, f))


if __name__ == '__main__':
    gold_file = sys.argv[1]
    auto_file = sys.argv[2]

    gold_tuples = readin_tuples(gold_file)
    auto_tuples = readin_tuples(auto_file)

    unlabelled_eval(gold_tuples, auto_tuples)
