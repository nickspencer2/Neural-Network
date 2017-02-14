# Neural Network
This is a Neural Network class I created in Python (version 2.X) along with some test data. It also uses numpy.
For instructions installing NumPy, visit this page: http://scipy.org/install.html .

In order to use the Neural Network, use the class I've provided in nn_generalized_nparrays.py called "NeuralNetwork".
The functions for the class are as follows:
NOTE: for an example of how to use the Neural Network, run the program using the test data in nn_generalized_nparrays.py .

- initialization function (arguments passed to the object when creating a Neural Network):
  - num_inputs: the number of "Neurons" to be in the first layer, AKA the number of inputs to the Neural Network
  - layer_sizes: the number of "Neurons" to be in the hidden layers and the output layer, in the form of a list
  - layer_neuron_weights (not required) (default = random --> [np.random.randn()]): the weights to be used for the layers. 
    This should be a 3-dimensional list, where the first index is the layer index, the second index is the neuron index, and
    the third index is the weight index that is incoming to that neuron.
  - layer_neuron_biases (not required) (default = random --> [np.random.randn()]): the biases to be used for the layers. 
    Same indexing scheme as layer_neuron_weights.
  - learningrate (not required) (default = 0.5): the learning rate to use when training the Neural Network.

- minibatchtrain (used to train the Neural Network using the minibatch method):
  - training data: the input to use for training data, this should be in the form of an array data structure (numpy array is reccomended)
    with the first index being the index identifying the specific data input instance, and the second index being the input to the specific
    neuron.
  - minibatch size: the size to use for a minibatch. A minibatch can be summarized as multiple instances of the training data used as a set
    to more efficiently train the Neural Network.
  - epochs: the number of epochs to run. An epoch can be summarized as one run-through of the training_data, while the next epoch will
    keep the same weights and biases in order to continue improving the Neural Network.
  - test_data (not required) (default = None; it should be noted that this will result in no accuracy reporting): the data used to 
    test the Neural Network's accuracy.
  - log (not required) (default = 0; this will result in no reporting of the progress): The frequency of printing out the progress
    of the Neural Network training. For example, if log = 5, then after 5 minibatches have been processed, the program will output an update.

- train (used to train the Neural Network with individual training instances):
  - training_inputs: the inputs used to train the Neural Network
  - training_outputs: the corresponding outputs used to train the Neural Network
  
- totalerror (used in conjunction with train to find the error in the Neural Network):
  - training_sets: the training inputs and outputs used to find the error the Neural Network outputs
