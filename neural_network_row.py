import numpy as np
import h5py

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
def d_sigmoid(x):
    y = sigmoid(x)
    return y*(1.0-y)
def tanh(x):
    return 2.0/(1+np.exp(-2*x)) - 1.0
def d_tanh(x):
    y = tanh(x)
    return 1.0 - y*y

def check_shape(shape):
    if len(shape)<2:
        raise ValueError('shape must be greater than 1',shape)
    for i in shape:
        if type(i) != int or i<1:
            raise ValueError('each value in shape must be an int and positive', shape)

class NeuralNetwork:
    '''
    multi layer neural network, the input and output values are treated as row vector.
    so the shape of a weight from l1 to l2 is size(l1) x size(l2)
    Note *** This notation is inverse to the neural_network.py
    '''
    def __init__(self, shape, fn=(sigmoid, d_sigmoid), learnrate=.2, epochs=5000):
        self.fn = fn[0]
        self.dfn= fn[1]
        self.learnrate = learnrate
        self.epochs = epochs

        self.weights = []
        self.bias   = []
        if shape != None:
            self._init(shape)
        self.debug = {}
        self.debug['progress'] = True
        self.debug['n'] = 50

    def _init(self, shape):
        '''
        init the weights and bias with random number in [0.0, 1.0)
        '''
        check_shape(shape)
        m = shape[0]
        for n in shape[1:]:
            w = np.random.random((m,n))
            b = np.random.random(n)
            self.weights.append(w)
            self.bias.append(b)
            m = n

    def init(self, shape, force=False):
        if len(self.weights)+len(self.bias)>0 and not force:
            raise Exception('init twice')
        self._init(shape)

    def shape(self):
        if len(self.weights)==0:
            return None
        weights = self.weights
        shape=[weights[0].shape[0]]
        for w in weights:
            shape.append(w.shape[1])
        return tuple(shape)

    def dump(self):
        print('#############################')
        print('layers : ', self.layers())
        print('learnrate = ', self.learnrate)
        print('epochs = ', self.epochs)
        print('fn = ', self.fn.__name__)
        print('dfn = ', self.dfn.__name__)
        print('weights : ')
        for w in self.weights:
            print(w)
        print('bias : ')
        for b in self.bias:
            print(b)


    def layers(self):
        return len(self.weights)
    def predict(self, X):
        '''
        predict single sample
        '''
        # assert len(X) == self.weights[0].shape[1]
        for w,b in zip(self.weights, self.bias):
            X = self.fn(np.dot(X, w) + b)
        return X

    # train the network, no batching
    def train(self, test_x, test_y):
        epochs = self.epochs
        learnrate = self.learnrate
        N = self.layers()+1
        show = self.debug['progress']
        tn = self.debug['n']
        for _ in range(epochs):
            if show and _ % tn == 0:
                print('epochs : %d/%d' % (_, epochs))
            for x, y in zip(test_x, test_y):
                # cache each input linear combination of the activation function
                hs = []
                xs = []
                for w,b in zip(self.weights, self.bias):
                    xs.append(x)
                    h = np.dot(x, w) + b
                    hs.append(h)
                    x = self.fn(h)
                error = y - x

                # backpropgation to adjust the weights
                acc = np.copy(error)
                for k in range(-1, -N, -1):
                    df = self.dfn(hs[k])
                    acc *= df
                    self.bias[k] += learnrate * acc
                    deltaW = xs[k][:, None] * acc
                    # R M => R !!!
                    acc = np.dot(self.weights[k], acc)
                    self.weights[k] += learnrate * deltaW

    def batch_train(test_x, test_y):
        '''
        batch training network
        '''
        epochs = self.epochs
        learnrate = self.learnrate
        N = self.layers() + 1
        for _ in range(epochs):
            delta_bias = [np.zeros(shape=b.shape) for b in self.bias]
            delta_weights = [np.zeros(shape=w.shape) for w in self.weights]
            for x, y in zip(test_x, test_y):
                # cache each input linear combination of the activation function
                hs = []
                xs = []
                for w,b in zip(self.weights, self.bias):
                    xs.append(x)
                    h = np.dot(x, w) + b
                    hs.append(h)
                    x = self.fn(h)
                error = y - x

                # backpropgation to adjust the weights
                acc = np.copy(error)
                for k in range(-1, -N, -1):
                    df = self.dfn(hs[k])
                    acc *= df
                    delta_bias[k] += acc
                    deltaW = xs[k][:, None] * acc
                    acc = np.dot(self.weights[k], acc)
                    delta_weights[k] += deltaW
                    

            m = len(test_x) # batch size
            for idx in range(len(self.weights)):
                self.weights[idx] += learnrate * delta_weights[idx] / m
            for idx in range(len(self.bias)):
                self.bias[idx] += learnrate * delta_bias[idx] / m

    def save(self, name):
        '''
        save this neural-network to disk file named `name`
        '''
        SaveNN(name, self)
    def load(self, name):
        '''
        load a neural-network from disk file named `name`
        **NOTE** this neural-network must not be initialized
        '''
        if len(self.weights)+len(self.bias)>0:
            raise Exception('load must do with an uninitialized network')
        return _LoadNN(name, self)

def SaveNN(name, net):
    f = h5py.File(name, 'w')
    n = net.layers()
    f.attrs['layers'] = n
    f.attrs['fn']  = net.fn.__name__
    f.attrs['dfn'] = net.dfn.__name__
    for i in range(n):
        f.create_dataset('weight%d' % i, data = net.weights[i])
        f.create_dataset('bias%d' % i, data = net.bias[i])
    f.attrs['learnrate'] = net.learnrate
    f.attrs['epochs']    = net.epochs
    # ignore debug info, currently
    f.close()

def LoadNN(name):
    return _LoadNN(name, NeuralNetwork(shape=None))

def _LoadNN(name, net):
    f = h5py.File(name, 'r')
    n = f.attrs['layers']
    weights, bias = [], []
    for i in range(n):
        d_weight, d_bias = f.get('weight%d' % i), f.get('bias%d' % i)
        weights.append(d_weight.value)
        bias.append(d_bias.value)
    net.weights = weights
    net.bias = bias
    net.fn = eval(f.attrs['fn'])
    net.dfn=eval(f.attrs['dfn'])
    net.learnrate = f.attrs['learnrate']
    net.epochs = f.attrs['epochs']

    f.close()
    return net

def _test_save_load(name):
    shapes = []
    shapes.extend([(1,1), (1,3), (2,4), (4,2), (3,1)])
    shapes.extend([(1,2,3,4), [2,3,4]])
    fn = [(sigmoid,d_sigmoid), (tanh, d_tanh)]
    idx=0
    for shape in shapes:
        f = fn[int(np.random.random()*len(fn))]
        learnrate = np.random.random()
        epochs = int(np.random.random()*200 + 100)
        net = NeuralNetwork(shape, fn=f, learnrate=learnrate, epochs=epochs)

        # start to dump
        net.dump()
        SaveNN(name + str(idx), net)

        net2 = LoadNN(name + str(idx))
        net2.dump()

        print('\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        idx += 1



if __name__ == '__main__':
    import time
    filename = time.strftime('test_save_load_%H-%M-%S.nn')
    _test_save_load(filename)
    net = NeuralNetwork(shape=None)
    net.load(filename)
    print(net)
