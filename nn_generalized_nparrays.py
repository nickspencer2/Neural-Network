#Nick Spencer 2016
import mnist_loader
import math
import numpy as np
import random

class Neuron:
    def __init__(self, num_inputs, bias=None, weights=None, outputneuron=False):
        self.outputneuron = outputneuron
        self.init_bias(bias)
        self.init_weights(num_inputs, weights)
    
    def init_weights(self, num_inputs, weights=None):
        if weights:
            self.weights = np.array([weights[i] if weights[i] else np.random.randn() for i in xrange(num_inputs)])
        else:
            self.weights = np.random.randn(num_inputs)
            
    def init_bias(self, bias=None):
        if bias:
            self.bias = bias
        else:
            self.bias = np.random.randn()
        
    def do_output(self, inputs):
        self.inputs = inputs
        self.output = self.activationfunction(self.netinput())
        return self.output
        
    def activationfunction(self, x):
        return 1/(1 + np.exp(-x))
        
    def netinput(self):
        return np.dot(self.inputs, self.weights) + self.bias
        
    def error(self, targetoutput):
        return ((self.output - targetoutput)**2)/2.0
        
    def dEdO(self, targetoutput):
        return self.output - targetoutput
        
    def dOdN(self):
        return self.output*(1 - self.output)
        
    def dNdW(self, i=None):
        if i:
            return self.inputs[i]
        else:
            return self.inputs
        
    def do_delta(self, targetoutput=None, nextlayer=None, hindex=0):
        if targetoutput != None:
            self.delta = np.dot(self.dEdO(targetoutput), self.dOdN())
        else:#hidden
            self.delta = np.dot(np.dot(nextlayer.getdeltas(), nextlayer.getweights().T[hindex]), self.dOdN())
        return self.delta
        
    def update(self, learningrate, batchsize=1):#updates using average nablas for the minibatch (given by batchsize)
        self.weights -= (learningrate*1.0/batchsize)*self.nablaws
        self.bias -= (learningrate*1.0/batchsize)*self.nablab
        
    def singleupdate(self, learningrate):
        self.weights -= learningrate*self.delta*self.dNdW()
        self.bias -= learningrate*self.delta
        
    def set_nablas(self, nablab=None, nablaws=None):
        if nablaws == None:#auto update using delta
            self.nablaws = self.delta*self.dNdW()
            self.nablab  = self.delta
        else:#set explicit value using input
            if len(nablaws) == 1:
                self.nablaws = np.full(len(self.weights), nablaws[0])
            else:
                self.nablaws = nablaws
            self.nablab = nablab
        return self.nablab, self.nablaws
        
    def incrementnablas(self, dnb, dnws):
        self.nablab += dnb
        self.nablaws = self.nablaws + dnws
        
        
            
class NeuronLayer:
    def __init__(self, size, num_inputs, weights=None, biases=None, outputlayer=False):
        self.size = size
        self.num_inputs = num_inputs
        self.outputlayer = outputlayer#true if this is an output layer
        self.init_neurons(biases, weights)
        
    def feedforward(self, inputs):
        self.inputs = inputs
        self.outputs = np.zeros(self.size)
        for i in xrange(len(self.neurons)):
            n = self.neurons[i]
            self.outputs[i] = n.do_output(inputs)
        return self.outputs
        
    def getdeltas(self):
        deltas = np.zeros(self.size)
        for i in xrange(len(self.neurons)):
            n = self.neurons[i]
            deltas[i] = n.delta
        self.deltas = deltas
        return deltas
        
    def dodeltas(self, targetoutputs=None, nextlayer=None):
        deltas = np.zeros(self.size)
        for i in xrange(len(self.neurons)):
            n = self.neurons[i]
            if targetoutputs != None:
                a = n.do_delta(targetoutput=targetoutputs[i])
                deltas[i] = a
            else:
                deltas[i] = n.do_delta(nextlayer=nextlayer, hindex=i)
        self.deltas = deltas
    
    def getweights(self):
        weights = np.zeros((self.size, len(self.neurons[0].weights)))
        for i in xrange(len(self.neurons)):
            n = self.neurons[i]
            weights[i] = n.weights
        self.weights = weights
        return weights
        
    def getinputs(self):
        inputs = np.zeros((self.zise, len(self.neurons[0].inputs)))
        for i in xrange(len(self.neurons)):
            n = self.neurons[i]
            inputs[i] = n.inputs
        self.inputs = inputs
        return inputs
        
    def init_neurons(self, biases, weights):
        self.neurons = []
        for i in xrange(self.size):
            if weights and biases:
                self.neurons.append(Neuron(num_inputs=self.num_inputs, bias=biases[i], weights=weights[i], outputneuron=self.outputlayer))
            elif weights:
                self.neurons.append(Neuron(num_inputs=self.num_inputs, weights=weights[i], outputneuron=self.outputlayer))
            elif biases:
                self.neurons.append(Neuron(num_inputs=self.num_inputs, bias=biases[i], outputneuron=self.outputlayer))
            else:
                self.neurons.append(Neuron(num_inputs=self.num_inputs, outputneuron=self.outputlayer))
                
    def donablas(self, nablabs=None, nablaws=None):
        self.nablabs = np.zeros(self.size)
        self.nablaws = np.zeros((self.size, len(self.neurons[0].inputs)))
        for i in xrange(len(self.neurons)):
            n = self.neurons[i]
            if nablabs != None and nablaws != None:
                self.nablabs[i], self.nablaws[i] = n.set_nablas(nablab=nablabs[i], nablaws=nablaws[i])#if None, updates using deltas, otherwise sets to value
            elif nablabs != None:
                self.nablabs[i], self.nablaws[i] = n.set_nablas(nablab=nablabs[i])
            elif nablaws != None:
                self.nablabs[i], self.nablaws[i] = n.set_nablas(nablaws=nablaws[i])
            else:
                self.nablabs[i], self.nablaws[i] = n.set_nablas()
            
    def getnablaws(self):
        nablaws = np.zeros((self.size, len(self.neurons[0].weights)))
        for i in xrange(len(self.neurons)):
            n = self.neurons[i]
            nablaws[i] = n.nablaws
        self.nablaws = nablaws
        return nablaws
            
    def getnablabs(self):
        nablabs = np.zeros(self.size)
        for i in xrange(len(self.neurons)):
            n = self.neurons[i]
            nablabs[i] = n.nablab
        self.nablabs = nablabs
        return nablabs

    def update(self, learningrate, batchsize=None, update_self=False):
        for i in xrange(len(self.neurons)):
            n = self.neurons[i]
            if batchsize:
                n.update(learningrate, batchsize)
            else:
                n.singleupdate(learningrate)
        if update_self:
            self.getdeltas()
            self.getinputs()
            self.getnablabs()
            self.getnablaws()
            self.getweights()
        
  
class NeuralNetwork:
    def __init__(self, num_inputs, layer_sizes, layer_neuron_weights=None, layer_neuron_biases=None, learningrate = 0.5):
        self.learningrate = learningrate
        self.num_inputs = num_inputs
        self.layer_sizes = layer_sizes
        self.init_layers(weights=layer_neuron_weights, biases=layer_neuron_biases)
        
    def init_layers(self, weights, biases):
        self.layers = []
        for i in xrange(len(self.layer_sizes)):
            if i == 0:#first hidden layer
                ni = self.num_inputs
            else:
                ni = self.layers[i-1].size
            if weights and biases:
                self.layers.append(NeuronLayer(size=self.layer_sizes[i], num_inputs=ni, weights=weights[i], biases=biases[i]))
            elif weights:
                self.layers.append(NeuronLayer(size=self.layer_sizes[i], num_inputs=ni, weights=weights[i]))
            elif biases:
                self.layers.append(NeuronLayer(size=self.layer_sizes[i], num_inputs=ni, biases=biases[i]))
            else:
                self.layers.append(NeuronLayer(size=self.layer_sizes[i], num_inputs=ni))
                
    def feedforward(self, inputs):
        out = inputs
        for layer in self.layers:
            out = layer.feedforward(out)
        return out
        
    def train(self, training_inputs, training_outputs):
        self.feedforward(training_inputs)
        for l in xrange(1, len(self.layers)+1):
            layer = self.layers[-l]
            print l
            if l == 1:#output layer
                layer.dodeltas(training_outputs)
            else:
                layer.dodeltas(nextlayer=self.layers[-l+1])
            layer.update(self.learningrate)
            
    def minibatchtrain(self, training_data, minibatchsize, epochs, test_data=None, log=0):
        if test_data:
            testdata_in  = [test_data[i][0] for i in xrange(len(test_data))]
            testdata_out = [test_data[i][1] for i in xrange(len(test_data))]
        for e in xrange(epochs):
            print "epoch: " + str(e)
            random.shuffle(training_data)
            minibatches = [training_data[k:k+minibatchsize] for k in xrange(0, len(training_data), minibatchsize)]
            for i in xrange(len(minibatches)):
                minibatch = minibatches[i]
                if log != 0:
                    if i%log == 0:
                        print "\tmini batch " + str(i) + " out of " + str(len(minibatches))
                self.minibatchupdate(minibatch, minibatchsize)
            if test_data:
                print "hits percentage: " + str(self.hitspercentage(testdata_in, testdata_out))
                
    def minibatchupdate(self, minibatch, minibatchsize):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.set_nablas(0, [0])
        minibatch_nbws = np.empty(minibatchsize, dtype=object)#indeces: training #, layer #, neuron #, weight #
        minibatch_nbbs = np.empty(minibatchsize, dtype=object)#indeces: training #, layer #, neuron/bias #
        i = 0
        for trainin, trainout in minibatch:
            self.feedforward(trainin)
            layer_nbws = np.empty(len(self.layers), dtype=object)#indeces: layer #, neuron #, weight #
            layer_nbbs = np.empty(len(self.layers), dtype=object)#indeces: layer #, neuron/bias #
            for l in xrange(1, len(self.layers)+1):
                layer = self.layers[-l]#going backwards
                if l == 1:#output layer
                    layer.dodeltas(trainout)
                    layer.donablas()
                else:
                    layer.dodeltas(nextlayer=self.layers[-l+1])
                    layer.donablas()
                layer_nbws[-l] = layer.getnablaws()
                layer_nbbs[-l] = layer.getnablabs()
            minibatch_nbws[i] = layer_nbws
            minibatch_nbbs[i] = layer_nbbs
            i += 1
        new_nbbs = np.sum(minibatch_nbbs, 0)
        new_nbws = np.sum(minibatch_nbws, 0)
        for i in xrange(len(new_nbws)):#i = layer #
            for j in xrange(len(new_nbws[i])):#j = neuron #
                n = self.layers[i].neurons[j]
                #n.set_nablas(np.array(new_nbbs[i][j]), np.array(new_nbws[i][j]))
                n.set_nablas(new_nbbs[i][j], new_nbws[i][j])
                n.update(learningrate=self.learningrate, batchsize=minibatchsize)
        
            
    def weightsum(self, weighttuple):
        return [sum(x) for x in zip(*weighttuple)]
            
    def totalerror(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feedforward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.layers[-1].neurons[o].error(training_outputs[o])
        return total_error
        
    def hitspercentage(self, test_in, test_out, log=False):
        hits = 0
        for i in xrange(len(test_in)):
            ti = test_in[i]
            to = test_out[i]        
            hits += self.hit(ti, to, log=log)
        return 100.0*(hits*1.0)/len(test_out)
        
    def hit(self, test_in, test_out, log=False):
        neuronoutputs = self.feedforward(test_in)
        answerind = np.argmax(neuronoutputs)
        if log:
            print "guess: " + str(answerind)
            print "answer: " + str(np.argwhere(test_out == 1))
        return int(answerind == np.argwhere(test_out == 1))

#XOR example              
#errors = []
#training_sets = [
#     [[0, 0], [0]],
#     [[0, 1], [1]],
#     [[1, 0], [1]],
#     [[1, 1], [0]]
# ]
#testnn = NeuralNetwork(len(training_sets[0][0]), 5, len(training_sets[0][1]))
#for i in xrange(10000):
#    training_inputs, training_outputs = random.choice(training_sets)
#    testnn.train(training_inputs, training_outputs)
#    e = testnn.totalerror(training_sets)
#    errors.append(e)
#    print i, round(e, 9)
#    
#if not sorted(errors):
#    print "errors worsened"
#else:
#    print "errors improved continuously"
                
                
                
                
##Meaningless numbers example               
#errors = []
#testnn = NeuralNetwork(2, [2, 2], layer_neuron_weights=[[[0.15, 0.2], [0.25, 0.3]], [[0.4, 0.45], [0.5, 0.55]]], layer_neuron_biases=[[.35]*2, [0.6]*2], learningrate=0.5)
#for i in range(10000):
#    testnn.train(np.array([0.05, 0.1]), np.array([0.01, 0.99]))
#    e = testnn.totalerror([[[0.05, 0.1], [0.01, 0.99]]])
#    print(i, e)
#    errors.append(e)
#if not sorted(errors):
#    print "errors worsened"
#else:
#    print "errors improved continuously"

#Trying out MNIST Data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = [(tr_d[0].reshape(784), tr_d[1].reshape(10)) for tr_d in training_data]
validation_data = [(vl_d[0].reshape(784), mnist_loader.vectorized_result(vl_d[1]).reshape(10)) for vl_d in validation_data]
test_data = [(ts_d[0].reshape(784), mnist_loader.vectorized_result(ts_d[1]).reshape(10)) for ts_d in test_data]
print "Data loaded."
testnn = NeuralNetwork(num_inputs=784, layer_sizes=[30, 10], learningrate=3.0)
testnn.minibatchtrain(training_data=training_data, minibatchsize=10, epochs=30, test_data=test_data, log=100)
print "Trained."