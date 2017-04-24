import codecs
import copy
from data_structures import Action, Node, State


def make_one_training_data(doc, vocab):
    """ Given a document in a conll-similar format,
    produce a list of (state, action).

    type doc: string
    param doc: '''
        SNT_LIST 
        snt1
        snt2
        ...
        EDGE_LIST 
        sntId_wordIdStart_wordIdEnd  annotation_label sntId_wordIdStart_wordIdEnd  link_label
        ...
    '''
        In EDGE_LIST, field 1 is child span, field 2 label for child,
        field 3 is parent span, field 4 is label for the link. 

        EDGE_LIST is sorted by field 1.
    """

    doc = doc.strip().split('\n')

    # create snt_list, edge_list
    snt_list = []
    edge_list = []
    mode = None
    for line in doc:
        if line.endswith('LIST'):
            mode = line.strip()
        elif mode == 'SNT_LIST':
            snt_list.append(line.strip())
        elif mode == 'EDGE_LIST':
            edge_list.append(line.strip().split())

    # add to vocab
    for snt in snt_list:
        for word in snt:
            if word not in vocab:
                vocab[word] = vocab.get(word, 0) + 1 

    # initialize state (put root node in the stack, initialize buffer)
    init_buffer = []
    for edge in edge_list:
        snt_id, word_id_start, word_id_end = edge[0].split('_')
        init_buffer.append(Node(snt_id, word_id_start, word_id_end, \
            '_'.join(snt_list[int(snt_id)].split()[
                int(word_id_start):int(word_id_end) + 1]), edge[1]))

    root_node = Node()
    state = State([root_node], init_buffer, snt_list)

    # for each edge in the edge_list, add a parent to 
    # the corresponding  node in the buffer 
    parent_count = {}
    for i, edge in enumerate(edge_list):
        snt_id, word_id_start, word_id_end = edge[0].split('_')
        child = Node(snt_id, word_id_start, word_id_end, '_'.join(
            snt_list[int(snt_id)].split()[
                int(word_id_start):int(word_id_end) + 1]))
        snt_id, word_id_start, word_id_end = edge[2].split('_')
        parent = Node(snt_id, word_id_start, word_id_end, '_'.join(
            snt_list[int(snt_id)].split()[
                int(word_id_start):int(word_id_end) + 1]))

        if not child.same_span(init_buffer[i]):
            print("ERROR! node in edge_list doesn't match node in buffer")
        else:
            init_buffer[i].parent = parent
            #print("parent added for: ", child, "; parent is: ", parent)
            parent_count[parent.ID] = parent_count.get(parent.ID, 0) + 1

    # create (state, action) list
    training_data = []  # a list of (state, action) tuples
    while state.buffer_:
        # shift
        training_data.append((copy.deepcopy(state), Action.SHIFT))
        state.shift()
        #print(Action.SHIFT)
        #print(state, end='\n\n')
        # reduce
        while len(state.stack) > 1:
            if state.stack[-1].parent.same_span(state.stack[0]) and \
                    parent_count.get(state.stack[-1].ID, 0) == 0: # root_reduce
                training_data.append((copy.deepcopy(state), Action.ROOT_REDUCE))
                parent_count[state.stack[0].ID] -= 1
                state.root_reduce()
                #print(Action.ROOT_REDUCE)
                #print(state, end='\n\n')
            elif state.stack[-1].parent.same_span(state.stack[-2]) and \
                    parent_count.get(state.stack[-1].ID, 0) == 0: # left_reduce
                training_data.append((copy.deepcopy(state), Action.LEFT_REDUCE))
                parent_count[state.stack[-2].ID] -= 1
                state.left_reduce()
                #print(Action.LEFT_REDUCE)
                #print(state, end='\n\n')
            elif state.stack[-2].parent and \
                    parent_count.get(state.stack[-2].ID, 0) == 0 and \
                    state.stack[-2].parent.same_span(state.stack[-1]): # right_reduce
                training_data.append((copy.deepcopy(state), Action.RIGHT_REDUCE))
                parent_count[state.stack[-1].ID] -= 1
                state.right_reduce()
                #print(Action.RIGHT_REDUCE)
                #print(state, end='\n\n')
            else: # no more reduces
                break

    return training_data


def make_training_data(train_file):
    """ Given a file of multiple documents in conll-similar format,
    produce a list of lists, each list is a list of (state, action) tuples;
    and the vocabulary of this training data set.
    """

    data = codecs.open(train_file, 'r', 'utf-8').read()
    doc_list = data.strip().split('\n\n')

    # a list of lists, each list is of (state, action) tuples
    training_data = []  
    count_vocab = {}

    for doc in doc_list:
        training_data.append(make_one_training_data(doc, count_vocab))

    # get the index_vocab from the count_vocab 
    index = 3
    vocab = {}
    for word in count_vocab:
        if count_vocab[word] > 0:   # cut off in frequent words
            vocab[word] = index 
            index += 1

    vocab.update({'<START>':0, '<STOP>':1, '<UNK>':2})

    return training_data, vocab


def make_one_test_data(doc):
    """ Given a document in a conll-similar format,
    produce an initial state. 

    type doc: string
    param doc: '''
        SNT_LIST 
        snt1
        snt2
        ...
        EDGE_LIST 
        sntId_wordIdStart_wordIdEnd  label sntId_wordIdStart_wordIdEnd  label
        ...
    '''
        In EDGE_LIST, field 1 is child span, field 2 label for child,
        field 3 is parent span, field 4 is label for the link. 
        Field 3 and field 4 are not used since this is test data.

        EDGE_LIST is sorted by field 1.
    """

    doc = doc.strip().split('\n')

    # create snt_list, edge_list
    snt_list = []
    edge_list = []
    mode = None
    for line in doc:
        if line.endswith('LIST'):
            mode = line.strip()
        elif mode == 'SNT_LIST':
            snt_list.append(line.strip())
        elif mode == 'EDGE_LIST':
            edge_list.append(line.strip().split())

    # initialize state (put root node in the stack, initialize buffer)
    init_buffer = []
    for edge in edge_list:
        snt_id, word_id_start, word_id_end = edge[0].split('_')
        init_buffer.append(Node(snt_id, word_id_start, word_id_end, \
            '_'.join(snt_list[int(snt_id)].split()[
                int(word_id_start):int(word_id_end) + 1]), edge[1]))

    root_node = Node()
    state0 = State([root_node], init_buffer, snt_list)

    return state0


def make_test_data(test_file):
    """ Given a file of multiple documents in conll-similar format,
    produce a list of states, each state is the initial state of 
    stack, buffer, and the snt_list for each document.
    """

    data = codecs.open(test_file, 'r', 'utf-8').read()
    doc_list = data.strip().split('\n\n')

    test_data = []

    for doc in doc_list:
        test_data.append(make_one_test_data(doc))

    return test_data
