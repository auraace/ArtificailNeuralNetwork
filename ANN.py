from random import random
from math import exp


def calculateNeuronOutput(nInput):
    return 1.0 / (1.0 + exp(-nInput))


def calculateNetInput(weights, bias, inputs):
    netInput = bias
    for i in range(len(weights)):
        netInput += weights[i] * inputs[i]
    return netInput


def generateOutput(network, inputs):
    for layer in network:
        tempInputs = []
        for neuron in layer:
            netInput = calculateNetInput(neuron['weights'], neuron['bias'], inputs)
            neuron['output'] = calculateNeuronOutput(netInput)
            tempInputs.append(neuron['output'])
        inputs = tempInputs
    return inputs


def errorFunction(output, coeff):
    return output * (1 - output) * coeff


def findErrors(network, real):
    #output layer
    layer = network[1]
    for i in range(len(layer)):
        neuron = layer[i]
        neuron['error'] = errorFunction(neuron['output'], real[i] - neuron['output'])

    #hidden layer
    layer = network[0]
    for i in range(len(layer)):
        resError = 0.0
        neuron = layer[i]
        for n in network[1]:
            resError += n['weights'][i] * n['error']
        neuron['error'] = errorFunction(neuron['output'], resError)


def updateNetwork(network, inputs, rate):
    expected = inputs.pop()
    for l in range(len(network)):
        layer = network[l]
        if l > 0:
            inputs = [neuron['output'] for neuron in network[l - 1]]
        for neuron in layer:
            for i in range(len(inputs)):
                neuron['weights'][i] += rate * neuron['error'] * inputs[i]
            neuron['bias'] += rate * neuron['error']

def randomWeights(num):
    weights = []
    for i in range(num):
        weights.append(random() - .5)
    return weights

def generateLayer(numInputs, numOutputs):
    layer = []
    for i in range(numOutputs):
        layer.append({'weights':randomWeights(numInputs), 'bias':random() - .5})
    return layer

#used to generate a random network
def generateNetwork(numInputs, numHidden, numOutputs):
    network = []
    hiddenLayer = generateLayer(numInputs, numHidden)
    network.append(hiddenLayer)
    outputLayer = generateLayer(numHidden, numOutputs)
    network.append(outputLayer)
    return network

def runANN(network, inputs, rate):
    for input in inputs:
        expected = input[-1]
        out = generateOutput(network, input)
        findErrors(network, expected)
        updateNetwork(network, input, rate)

        for layer in test:
            print(layer)


network = generateNetwork(2,3,2)
# network for part 1
test = [[{'weights': [0.1, -0.2], 'bias': 0.1}, {'weights': [0, 0.2], 'bias': 0.2}, {'weights': [0.3, -0.4], 'bias': 0.5}], [{'weights': [-0.4, 0.1, 0.6], 'bias': -0.1}, {'weights': [0.2, -0.1, -0.2], 'bias': 0.6}]]
rate = .1
inputs = [[0.6, 0.1, [1, 0]],
          [0.2, 0.3, [0, 1]]]
runANN(test, inputs, rate)
