import numpy as np

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

    def __init__(self, shape, fn=(sigmoid, d_sigmoid), learnrate=.2, epochs=5000):
        self.shape = shape
        self.fn = fn[0]
        self.dfn= fn[1]
        self.learnrate = learnrate
        self.epochs = epochs
        check_shape(shape)
        self._init()
        self.progress = True

    def _init(self):
        self.weights = []
        self.bias   = []
        shape = self.shape
        n = shape[0]
        for m in shape[1:]:
            w = np.random.random((m,n))
            b = np.random.random(m)
            self.weights.append(w)
            self.bias.append(b)
            n = m

    def layers(self):
        return len(self.weights)
    # predict single sample
    def predict(self, X):
        # assert len(X) == shape[0]
        for w,b in zip(self.weights, self.bias):
            X = self.fn(np.dot(w, X) + b)
        return X

    # train the network, no batching
    def train(self, test_x, test_y):
        epochs = self.epochs
        learnrate = self.learnrate
        N = len(self.shape)
        for _ in range(epochs):
            if self.progress and _ % 50 == 0:
                print('epochs : %d/%d' % (_, epochs))
            for x, y in zip(test_x, test_y):
                # cache each input linear combination of the activation function
                hs = []
                xs = []
                for w,b in zip(self.weights, self.bias):
                    xs.append(x)
                    h = np.dot(w, x) + b
                    hs.append(h)
                    x = self.fn(h)
                y2 = x
                delta = y - y2

                # backpropgation to adjust the weights
                acc = np.copy(delta)
                for k in range(-1, -N, -1):
                    df = self.dfn(hs[k])
                    acc *= df
                    self.bias[k] += learnrate * acc
                    # FIXME: treat acc as col vector and hs[k] as row vector
                    # so the result is a matrix which matches the shape of the weights
                    # !!! 
                    # deltaW = np.dot(acc, xs[k])
                    deltaW = np.dot(acc.reshape((-1,1)), xs[k].reshape((1,-1)))
                    # R M => R !!!
                    # acc = np.dot(acc, self.weights[k])
                    acc = np.dot(acc.reshape((1,-1)), self.weights[k])[0]
                    self.weights[k] += learnrate * deltaW

    def batch_train(test_x, test_y):
        '''
        batch training network
        '''
        pass

# it's perfect to use tanh
def test_xor():
    shape=(2,2,1)
    net = NeuralNetwork(shape, fn=(tanh, d_tanh))
    print(net.shape)
    i = 0
    for n in net.shape:
        print('layer%d  %d' % (i, n))
        i += 1
    for w in net.weights:
        print('weights shape = ', w.shape)
        
    for b in net.bias:
        print('bias length = ', len(b))

    test_x, test_y = [], []
    
    #for _ in range(3):
    #    test_x.append(np.random.random(shape[0])*20 -30)
    #    test_y.append(np.random.random(shape[-1]))
    test_x.append(np.array([0,0]))
    test_x.append(np.array([0,1]))
    test_x.append(np.array([1,0]))
    test_x.append(np.array([1,1]))
    test_y.append(np.array([0]))
    test_y.append(np.array([1]))
    test_y.append(np.array([1]))
    test_y.append(np.array([0]))
    net.learnrate = .2
    net.epochs = 5000
    net.progress = False
    net.train(test_x, test_y)
    
    for xx,yy in zip(test_x, test_y):
        y = net.predict(xx)
        print('xor', xx, 'predict=', y, 'target = ', yy)

# range 
# (cx,cy,r) ==(x,y)==> (length, in?)
center = np.array([2,3])
R = 4
L = 8
def teach(xx):
    p = xx - center
    l = np.sqrt(np.dot(p,p))
    if l<=R:
        yes = 1
    else:
        yes = 0
    l /= L
    return np.array([l, yes])
    
def test_circle():
    shape=(2,2,2,2,2,2,2)
    net = NeuralNetwork(shape, fn=(tanh, d_tanh))
    print(net.shape)
    i = 0
    for n in net.shape:
        print('layer%d  %d' % (i, n))
        i += 1
    for w in net.weights:
        print('weights shape = ', w.shape)
        
    for b in net.bias:
        print('bias length = ', len(b))

    test_x, test_y = [], []
    
    for _ in range(3000):
        radius = np.random.random()* L
        theta = np.random.random() * np.pi * 2
        tmp = np.array([np.cos(theta), np.sin(theta)]) * radius + center
        test_x.append(tmp)
        test_y.append(teach(tmp))

    net.learnrate = .2
    net.epochs = 3000
    net.train(test_x, test_y)
    
    dataX, dataY = [], []
    for _ in range(20):
        radius = np.random.random()* R
        theta = np.random.random() * np.pi * 2
        tmp = np.array([np.cos(theta), np.sin(theta)]) * radius + center
        dataX.append(tmp)
        dataY.append(teach(tmp))
        
    for w,b in zip(net.weights, net.bias):
        print(w,b)
    for xx,yy in zip(dataX, dataY):
        y = net.predict(xx)
        print((xx,yy), 'predict=', y, 'target = ', yy)

def test_not():
    test_x, test_y = [], []
    for i in [0,1]:
        test_x.append(np.array([i]))
        test_y.append(np.array([1-i]))
    net = NeuralNetwork(shape=(1,1), fn=(tanh, d_tanh))
    #net=NeuralNetwork(shape=(1,1))

    net.learnrate = .2
    net.epochs = 8000
    net.progress = False
    net.train(test_x, test_y)
    for x,y in zip(test_x,test_y):
        print('not: ', x, 'predict=', net.predict(x), 'target=', y)

def test_and_or(test_x, test_y, tag=None):
    
    #net = NeuralNetwork(shape=(2,1), fn=(tanh, d_tanh))
    net = NeuralNetwork(shape=(2,1))
    
    net.learnrate = .2
    net.epochs = 8000
    net.progress = False
    net.train(test_x, test_y)
    for x,y in zip(test_x,test_y):
        print(tag, x, 'predict=', net.predict(x), 'target=', y)
def test_and():
    test_x, test_y = [], []
    for i in [0,1]:
        for j in [0,1]:
            test_x.append(np.array([i,j]))
            test_y.append(np.array([0]))
    test_y[-1][0] = 1
    test_and_or(test_x, test_y, tag='and: ')
    
def test_or():
    test_x, test_y = [], []
    for i in [0,1]:
        for j in [0,1]:
            test_x.append(np.array([i,j]))
            test_y.append(np.array([1]))
    test_y[0][0] = 0
    test_and_or(test_x, test_y, tag='or: ')
    
if __name__ == '__main__':
    #test_xor()
    #test_circle()
    test_and()
    test_or()
    test_not()
    test_xor()
