import copy


class Action:
    SHIFT = 'shift' 
    LEFT_REDUCE = 'left_reduce'
    RIGHT_REDUCE = 'right_reduce'
    ROOT_REDUCE = 'root_reduce'


class Node:
    def __init__(self, snt_id='-1', word_id_start='-1', word_id_end='-1', words='root'):
        self.snt_id = snt_id
        self.word_id_start = word_id_start
        self.word_id_end = word_id_end
        self.ID = '_'.join([snt_id, word_id_start, word_id_end])
        self.words = words
        self.parent = None
        self.children = []

    def same_span(self, node):
        if node.snt_id == self.snt_id and \
            node.word_id_start == self.word_id_start and \
            node.word_id_end == self.word_id_end:
                return True
        return False

    def __str__(self):
        return '\t'.join([str(self.snt_id), str(self.word_id_start),
                str(self.word_id_end), self.words])


class State:
    def __init__(self, stack=[], buffer_=[], snt_list=[]):
        self.stack = stack 
        self.buffer_ = buffer_
        self.snt_list = snt_list

    def shift(self):
        self.stack.append(self.buffer_[0])
        self.buffer_ = self.buffer_[1:]

    def left_reduce(self):
        self.stack[-1].parent = self.stack[-2]
        self.stack[-2].children.append(self.stack[-1])
        self.stack.pop()

    def right_reduce(self):
        self.stack[-2].parent = self.stack[-1]
        self.stack[-1].children.append(self.stack[-2])
        parent_node = self.stack[-1]
        self.stack.pop()
        self.stack.pop()
        self.stack.append(parent_node)

    def root_reduce(self):
        self.stack[-1].parent = self.stack[0]
        self.stack[0].children.append(self.stack[-1])
        self.stack.pop()

    def __str__(self):
        tmp = 'stack: ['
        for node in self.stack:
            tmp += node.words + ', '
        tmp = tmp[:-2] + ']\nbuffer: ['
        for span in self.buffer_:
            tmp += span.words + ', '
        tmp = tmp[:-2] if not tmp.endswith('buffer: [')else tmp 
        tmp += ']'
        return tmp
