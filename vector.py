import codecs


class Vector:
    def __init__(self, dic):
        self.v = dic

    def __iadd__(self, other):
        for key in other.v:
            if key in self.v:
                self.v[key] += other.v[key]
            else:
                self.v[key] = other.v[key]
        return self

    def __isub__(self, other):
        for key in other.v:
            if key in self.v:
                self.v[key] -= other.v[key]
            else:
                self.v[key] = -other.v[key]
        return self

    def __rmul__(self, other):
        result = Vector()
        for key in self.v:
            result.v[key] = self.v[key] * other
        return result

    def dot(self, other):
        sum_ = 0
        for key in other.v:
            if key in self.v:
                sum_ += self.v[key] * other.v[key]
        return sum_

    def __str__(self):
        tmp = ''
        for key in self.v:
            tmp += key + '\t' + str(self.v[key]) + '\n'
        return tmp 

    def save(self, filename):
        with codecs.open(filename, 'w', 'utf-8') as f:
            for key in self.v:
                f.write(key + '\t' + str(self.v[key]) + '\n')

    def load(self, filename):
        with codecs.open(filename, 'r', 'utf-8') as f:
            lines = [line.strip().split('\t') for line in f.readlines()]
            self.v = {line[0]:float(line[-1]) for line in lines}
