import numpy as np
import pandas as pd

# Print iterations progress
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


def relu(x):
    return max(0,x);

def relu_prime(fx):
    return (fx != 0) * 1

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(fx):
    return fx * (1 - fx)
    
def tanh(x):
    return 2 / (1+np.exp(-2*x)) - 1
    
def tanh_prime(fx):
    return 1 - fx ** 2

def identity(x):
    return x

def identity_prime(x):
    return 1

def mse(truth, prediction):
    return (prediction - truth) ** 2

def mse_derivative(truth, prediction):
    return 2 * (prediction - truth)

activation_functions = {
    "relu" : np.vectorize(relu),
    "sigmoid" : np.vectorize(sigmoid),
    "tanh" : np.vectorize(tanh),
    "identity" : np.vectorize(identity)
}

activation_derivatives = {
    "relu" : np.vectorize(relu_prime),
    "sigmoid" : np.vectorize(sigmoid_prime),
    "tanh" : np.vectorize(tanh_prime),
    "identity" : np.vectorize(identity_prime)
}

cost_functions = {
    "mse" : np.vectorize(mse)
}

cost_derivatives = {
    "mse" : np.vectorize(mse_derivative)
}

class Layer:
  
    def __init__(self, input_size, output_size, activation_function, activation_derivative = None, bias = True, last_layer_bias = True, dropout = 1):
        """Initializes a layer to be densely connected into a neural network.
        
        Args:
          - size: the number of neurons in the layer (does not include the bias if there is one)
          - activation_function: the activation function to be applied to the weighted sum. Must pass either a string or a np.vectorized function.
          - activation_derivative: if passed a non-string to activation_function, this must be a np.vectorized function as well
          - bias: determines is a bias node will be present in this layer or not     
          - last_layer_bias: determines if the layer that this layer connects to had a bias present or not
          - dropout: a proportion from 0 to 1 inclusive that determines how likely it is for a single node to be dropped out
        """
        if isinstance(activation_function, str):
            self.activation_function = activation_functions[activation_function]
            self.activation_derivative = activation_derivatives[activation_function]
        elif activation_derivative is None:
            raise ParameterError
        else:
            self.activation_function = activation_function
            self.activation_derivative = activation_derivative
            
        self.bias = bias
        self.input_size = input_size
        self.output_size = output_size+1 if bias else output_size
        # These are the weights that connect the inputs to this layer's output neurons
        self.weights = np.random.normal(loc = 0.0, scale = (2 / input_size)**0.5, size = (self.input_size, output_size)) # NOT self.output_size
        self.inputs = None
        self.last_layer_bias = last_layer_bias
        self.dropout = dropout
    
    def forward_prop(self, inputs, for_train = False):
        self.inputs = inputs

        activated_values = self.activate(inputs.dot(self.weights))
        
        if for_train and self.dropout < 1:
            activated_values *= (np.random.rand(*activated_values.shape) > self.dropout) / self.dropout

        self.past = activated_values

        return activated_values 
                
    def activate(self, weighted_sums):
        """Passes a matrix of weighted sums through an activation function. Adds a bias column to the end if wanted.
        
        Args:
          - weighted_sums: a matrix composed of all nodes of prior layer multiplied by their synapses and summed up
          
        Returns:
          A matrix of the same size (plus one column if bias is wanted) where each value has been activated.
        """
        activated_values = self.activation_function(weighted_sums)
        
        if self.bias:
            bias_column = np.ones((activated_values.shape[0],1))
            activated_values = np.append(activated_values, bias_column, axis = 1)
                
        return activated_values
    
    def update_weights(self, dldh_prod, eta, l2_value):
        """Update the weights based on the partial derivatives of the outputs of the nodes within this layer.
        
        Args:
          - dldh_prod: a vector containing the partial derivatives of each node multiplied by the activation function's derivative (excluding the bias node)
        """
        
        dldw = self.inputs.T.dot(dldh_prod)
        if self.last_layer_bias:
            self.weights[:-1, :] = (1 - eta * l2_value) * self.weights[:-1, :] - eta * dldw[:-1, :]
            self.weights[-1,:] = self.weights[-1,:] - eta * dldw[-1,:] # Biases are not affected by regularization
        else:
            self.weights = (1 - eta * l2_value) * self.weights - eta * dldw
    
    def back_prop(self, dldh, eta, l2_value):
        """Calculate the partial derivatives of the prior layer and signal to update this layer's weights.
        
        Args:
          - dldh: a matrix of shape [observations, number of nodes in layer] containing the partial derivatives of each node (including the bias node)
        
        Return:
          A matrix for the prior layer containing the partial derivatives of each of their nodes.
        """
        past = self.past
        if self.bias: #remove bias
            dldh = dldh[:, :-1]
            past = past[:, :-1]
            
        activation_derivatives = self.activation_derivative(past)
        dldh_prod = np.multiply(activation_derivatives, dldh)
        
        prior_dldh = dldh_prod.dot(self.weights.T)
        
        self.update_weights(dldh_prod, eta, l2_value)
        
        return prior_dldh
    
class NeuralNetwork:
    
    def __init__(self, input_units, cost = "mse", input_bias = True):
        """Initializes a neural network model with no hidden layers. Must add them manually to define one completely.
        
        Args: 
          - input_units: number of neurons in the input layer
        """
        self.layers = []
        self.dims = [input_units+1 if input_bias else input_units]
        self.cost = cost_functions[cost]
        self.cost_derivative = cost_derivatives[cost]
        self.input_bias = input_bias
        
    def add_layer(self, nodes, activation_function = "relu", bias = True, dropout = 1):
        """Adds a layer to the network. Assumes it is to be fully connected.
        
        Args:
          - nodes: the size of the layer
          - activation_function: the activation function specified via string
          - bias: whether to have a bias or not
        
        Returns:
          A reference to this object to chain method calls.
        """
        if len(self.layers) > 0:
            last_layer_bias = self.layers[-1].bias
        else:
            last_layer_bias = self.input_bias
            
        layer = Layer(self.dims[-1], nodes, activation_function, bias = bias, last_layer_bias = last_layer_bias, dropout = dropout)
        self.layers.append(layer)
        self.dims.append(layer.output_size)
                
        return self
    
    def count_weights(self):
        return sum(layer.input_size * layer.output_size for layer in self.layers)
        
    def predict(self, x, for_train = False):
        """Predicts output for the given input x.
        
        Args:
          - x: an array of length `input_units`
          
        Returns:
          Predicted output for the given input.
        """
        
        # assume x is a horizontal vector
        output = x
        
        if self.input_bias:
            output = np.append(output, np.ones((x.shape[0], 1)), axis = 1)
        
        for layer in self.layers:
            output = layer.forward_prop(output)
        
        return output
        
    def process_input(self, z):
        if z is None:
            return z
            
        if isinstance(z, pd.Series):
            z = pd.DataFrame(z)
        
        if isinstance(z, pd.DataFrame):
            z = z.astype(float).as_matrix()
            
        if len(z.shape) <= 1:
            z = z.reshape((z.shape[0], 1))
        
        return z

        
    def fit(self, X, Y, test_X = None, test_Y = None, eta = 0.001, epochs = 10, batch_size = 32, verbose = True, l2_value = 0):
        """Trains the network based on the input data against the truth given.
        
        Args:
          - X: a matrix or dataframe of shape [data points, input features]
          - Y: a matrix or dataframe of shape [data points, output features]
          - eta: the learning rate
          - epochs: number of times to iterate over the entire dataset
          - batch_size: the number of data points to step through before updating the weights
        """
        
        X = self.process_input(X)
        Y = self.process_input(Y)
        test_X = self.process_input(test_X)
        test_Y = self.process_input(test_Y)
        
        print("Inputs have been converted. Training starting now.")
                
        for epoch in range(epochs):
            if verbose:
                predicted_y = self.predict(X)
                train_costs = self.cost(Y, predicted_y).mean()
                if test_X is None or test_Y is None:
                    print("Epoch", epoch, "\tTraining cost:", train_costs)
                else:
                    predicted_y = self.predict(test_X)
                    test_costs = self.cost(test_Y, predicted_y).mean()
                    print("Epoch", epoch, "\tTraining cost:", train_costs, "\tTesting cost:", test_costs)
                    
            for i in range(0, X.shape[0], batch_size):
                printProgressBar(i, X.shape[0], prefix = "Epoch " + str(epoch))
                end_point = min(i + batch_size, X.shape[0])
                self.update(X[i:end_point,:], Y[i:end_point], eta, X.shape[0], l2_value)
            printProgressBar(X.shape[0], X.shape[0], prefix = "Epoch " + str(epoch))
            
        if verbose:
            predicted_y = self.predict(X)
            train_costs = self.cost(Y, predicted_y).mean()
            if test_X is None or test_Y is None:
                print("Epoch", epoch, "\tTraining cost:", train_costs)
            else:
                predicted_y = self.predict(test_X)
                test_costs = self.cost(test_Y, predicted_y).mean()
                print("Epoch", epoch, "\tTraining cost:", train_costs, "\tTesting cost:", test_costs)
       
        
    def update(self, x, y, eta, n, l2_value):
        """Updates neural network weights based on new training data.
        
        Args:
          - x: a matrix of shape [observations, input_features]
          - y: a matrix of shape [observations, output_features]
          - eta: a float representing the learning rate
          - n: an integer representing the total amount of training observations in the entire dataset
        """
        prediction = self.predict(x, for_train = True)
        dldh = self.cost_derivative(y, prediction)
        for i, layer in enumerate(self.layers[::-1]):
            dldh = layer.back_prop(dldh, eta / x.shape[0], l2_value / n)
            # divide eta by the batch_size so as to account for multiple updates at the same time
            # divide lambda by the total samples for L2 regularization
            
