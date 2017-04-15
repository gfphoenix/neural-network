import numpy as np
import neural_network as nn
#from neural_network import NeuralNetwork
from neural_network import *

zero, one = np.array([0]), np.array([1])
test_x_and = [np.array([0,0]), np.array([0,1]), np.array([1,0]), np.array([1,1])]
test_y_and = [zero, zero, zero, one]
test_x_or  = test_x_and
test_y_or  = [zero, one, one, one]
test_x_not = [zero, one]
test_y_not = [one, zero]
test_x_xor = test_x_and
test_y_xor = [zero, one, one, zero]

# it's perfect to use tanh
def test_xor():
    shape=(2,2,1)
    net = NeuralNetwork(shape, fn=(nn.tanh, nn.d_tanh))
    for w in net.weights:
        print('weights shape = ', w.shape)
    for b in net.bias:
        print('bias length = ', len(b))

    test_x, test_y = [], []

    net.debug['progress'] = False
    net.train(test_x_xor, test_y_xor)

    for xx,yy in zip(test_x_xor, test_y_xor):
        y = net.predict(xx)
        print('xor', xx, 'predict=', y, 'target = ', yy)
    net.save('xor-2-2-1.nn')

def test_not():
    net = NeuralNetwork(shape=(1,1), fn=(tanh, d_tanh))
    #net=NeuralNetwork(shape=(1,1))

    net.learnrate = .2
    net.epochs = 8000
    net.debug['progress'] = False
    net.train(test_x_not, test_y_not)
    for x,y in zip(test_x_not, test_y_not):
        print('not: ', x, 'predict=', net.predict(x), 'target=', y)
    net.save('not-1-1.nn')

def test_and_or(test_x, test_y, tag=None):
    #net = NeuralNetwork(shape=(2,1), fn=(tanh, d_tanh))
    net = NeuralNetwork(shape=(2,1))

    net.learnrate = .2
    net.epochs = 8000
    net.debug['progress'] = False
    net.train(test_x, test_y)
    for x,y in zip(test_x,test_y):
        print(tag, x, 'predict=', net.predict(x), 'target=', y)
    return net

def test_and():
    net = test_and_or(test_x_and, test_y_and, tag='and: ')
    net.save('and-2-1.nn')

def test_or():
    net = test_and_or(test_x_or, test_y_or, tag='or: ')
    net.save('or-2-1.nn')

def init():
    test_xor()
    test_and()
    test_or()
    test_not()

def update(name, test_x, test_y, tag):
    net = LoadNN(name)
    net.learnrate *= .9
    net.debug['progress'] = False
    net.train(test_x, test_y)
    for x,y in zip(test_x, test_y):
        y2 = net.predict(x)
        print(tag, x, y, '==>', y2)
    net.save(name)

def train():
    update('xor-2-2-1.nn', test_x_xor, test_y_xor, 'xor')
    update('and-2-1.nn', test_x_and, test_y_and, 'and')
    update('or-2-1.nn', test_x_or, test_y_or, 'or')
    update('not-1-1.nn', test_x_not, test_y_not, 'not')

if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        init()
