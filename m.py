import numpy as np
import neural_network as nn
#from neural_network import NeuralNetwork
from neural_network import *

# (cx,cy,r) ==(x,y)==> (length, in?)
center = np.array([0,0])
R = 4
L = 8
def teach(xx):
    p = xx - center
    if np.dot(p,p)<=R*R:
        yes = 1
    else:
        yes = 0
    return np.array([yes])

def make_circle_data(n):
    x,y = [],[]
    for _ in range(n):
        radius = np.random.random()* L
        theta = np.random.random() * np.pi * 2
        tmp = np.array([np.cos(theta), np.sin(theta)]) * radius + center
        x.append(tmp)
        y.append(teach(tmp))
    return x,y

def test_circle():
    name = 'circle.nn'
    try:
        net = LoadNN(name)
        net.learnrate *= .6
        print('learnrate=', net.learnrate)
        net.debug['n'] = 500
    except Exception as e:
        shape=(2,512,2,1)
        net = NeuralNetwork(shape, fn=(tanh, d_tanh))
        for w in net.weights:
            print('weights shape = ', w.shape)
        for b in net.bias:
            print('bias length = ', len(b))

    net.learnrate = .2
    net.epochs = 3000
    test_x, test_y = make_circle_data(500)
    net.train(test_x, test_y)

    m = 40
    delta = [0,0]
    dataX, dataY = make_circle_data(m)
    for xx,yy in zip(dataX, dataY):
        err = yy - net.predict(xx)
        print(xx,yy, 'error=', err)
        delta[0] += err[0] * err[0]
    delta[0] /= m
    print('error = ', delta)
    net.save(name)

# two lines
k=[-.2, -1]
b=[.5, .8]
def line1(x):
    return k[0]*x + b[0]
def line2(x):
    return k[1] * x + b[1]
def test_2line():
    shape = (2,2,2,1)
    net = NeuralNetwork(shape, fn=(tanh, d_tanh))
    #net=NeuralNetwork(shape=(1,1))
    net.learnrate = .2
    net.epochs = 5000
    #net.debug['progress'] = False
    test_x,test_y = [], []
    for _ in range(1000):
        pos = np.random.random(2)
        y1 = line1(pos[0])
        y2 = line2(pos[0])
        test_x.append(pos)
        if pos[1] >= y1 and pos[1] >= y2:
            test_y.append(np.array([1.0]))
        else:
            test_y.append(np.array([0.0]))

    net.train(test_x, test_y)

    n = 60
    for _ in range(n):
        x = np.random.random(2)
        y = 0.0
        if x[1] >= line1(x[0]) and x[1] >= line2(x[0]):
            y = 1.0
        y2 = net.predict(x)
        print(x, y, y2)

    SaveNN('two_line.nn', net)

def test_load_2line():
    name = 'two_line.nn'
    net = LoadNN(name)
    net.debug['progress'] = False
    net.learnrate *= .5
    n_train = 100
    n_test = 20
    test_x, test_y = [], []
    for _ in range(n_train + n_test):
        pos = np.random.random(2)
        y1 = line1(pos[0])
        y2 = line2(pos[0])
        test_x.append(pos)
        if pos[1] >= y1 and pos[1] >= y2:
            test_y.append(np.array([1.0]))
        else:
            test_y.append(np.array([0.0]))

    net.train(test_x[n_test:], test_y[n_test:])

    for x,y in zip(test_x[:n_test], test_y[:n_test]):
        y2 = net.predict(x)
        print(x, y, y2)

    SaveNN(name, net)

def make_length_data(n):
    x,y = [],[]
    for _ in range(n):
        pos = np.random.random(2)*4
        x.append(pos)
        l = np.sqrt(np.dot(pos,pos))/5.7
        y.append(np.array([l]))
    return x,y

def test_length():
    name = 'length.nn'
    try:
        net = LoadNN(name)
        net.learnrate *= .9
        print('learnrate=', net.learnrate)
        #net.debug['n'] = 500
    except Exception as e:
        shape=(2,3,2,1)
        net = NeuralNetwork(shape, fn=(tanh, d_tanh))

    net.epochs = 500
    test_x, test_y = make_length_data(1000)
    net.train(test_x, test_y)

    m = 40
    delta = [0,0]
    dataX, dataY = make_length_data(m)
    for xx,yy in zip(dataX, dataY):
        err = yy - net.predict(xx)
        print(xx,yy, 'error=', err)
        delta[0] += err[0] * err[0]
    delta[0] /= m
    print('error = ', delta)
    net.save(name)

if __name__ == '__main__':
    #test_circle()
    #test_2line()
    #test_load_2line()
    #test_circle()
    test_length()
