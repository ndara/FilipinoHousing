import numpy as np
import pandas as pd

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
    return ((prediction - truth) ** 2).mean()

def mse_derivative(truth, prediction):
    return (2 * (prediction - truth)).mean()

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
    "mse" : mse
}

cost_derivatives = {
    "mse" : mse_derivative
}

class Layer:
  
    def __init__(self, input_size, output_size, activation_function, activation_derivative = None, bias = True):
        """Initializes a layer to be densely connected into a neural network.
        
        Args:
          - size: the number of neurons in the layer (does not include the bias if there is one)
          - activation_function: the activation function to be applied to the weighted sum. Must pass either a string or a np.vectorized function.
          - activation_derivative: if passed a non-string to activation_function, this must be a np.vectorized function as well
          - bias: determines is a bias node will be present in this layer or not        
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
        self.weights = np.random.normal(loc = 0.0, scale = (2 / input_size)**0.5, size = (self.input_size, output_size)) # NOT self.output_size
        self.inputs = None
    
    def forward_prop(self, inputs):
        self.inputs = inputs
        return self.activate(inputs.dot(self.weights))
        
    def activate(self, weighted_sums):
        """Passes a matrix of weighted sums through an activation function. Adds a bias column to the end if wanted.
        
        Args:
          - weighted_sums: a matrix composed of all nodes of prior layer multiplied by their synapses and summed up
          
        Returns:
          A matrix of the same size (plus one column if bias is wanted) where each value has been activated.
        """
        activated_values = self.activation_function(weighted_sums)
        if self.bias:
            bias_column = np.ones((weighted_sums.shape[0],1))
            activated_values = np.append(activated_values, bias_column, axis = 1)
        
        self.past = activated_values
        
        return activated_values
    
    def update_weights(self, dldh_prod, eta):
        """Update the weights based on the partial derivatives of the outputs of the nodes within this layer.
        
        Args:
          - dldh_prod: a vector containing the partial derivatives of each node multiplied by the activation function's derivative (excluding the bias node)
        """
        
        dldw = self.inputs.T.dot(dldh_prod)
        self.weights -= eta * dldw
        
    
    def back_prop(self, dldh, eta):
        """Calculate the partial derivatives of the prior layer and signal to update this layer's weights.
        
        Args:
          - dldh: a vector containing the partial derivatives of each node (including the bias node)
        
        Return:
          A vector for the prior layer containing the partial derivatives of each of their nodes.
        """
        past = self.past
        if self.bias: #remove bias
            dldh = dldh[:, :-1]
            past = past[:, :-1]
            
        activation_derivatives = self.activation_derivative(past)
        dldh_prod = np.multiply(activation_derivatives, dldh)
        self.update_weights(dldh_prod, eta)
        
        prior_dldh = self.weights.dot(dldh_prod.T).T
        
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
        
    def add_layer(self, nodes, activation_function = "relu", bias = True):
        """Adds a layer to the network. Assumes it is to be fully connected.
        
        Args:
          - nodes: the size of the layer
          - activation_function: the activation function specified via string
          - bias: whether to have a bias or not
        
        Returns:
          A reference to this object to chain method calls.
        """
        layer = Layer(self.dims[-1], nodes, activation_function, bias = bias)
        self.layers.append(layer)
        self.dims.append(layer.output_size)
                
        return self
    
    def count_weights(self):
        return sum(layer.input_size * layer.output_size for layer in self.layers)
        
    def predict(self, x):
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
        
    def fit(self, X, Y, eta, epochs, batch_size = 1, verbose = True):
        """Trains the network based on the input data against the truth given.
        
        Args:
          - X: a matrix or dataframe of shape [data points, input features]
          - Y: a matrix or dataframe of shape [data points, output features]
          - eta: the learning rate
          - epochs: number of times to iterate over the entire dataset
          - batch_size: the number of data points to step through before updating the weights
        """
        
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X)
        
        if isinstance(X, pd.DataFrame):
            X = X.astype(float).as_matrix()
        
        if isinstance(Y, pd.Series):
            Y = pd.DataFrame(Y)
        
        if isinstance(Y, pd.DataFrame):
            Y = Y.astype(float).as_matrix()
        
        for epoch in range(epochs):
            if verbose:
                predicted_y = self.predict(X)
                costs = self.cost(Y, predicted_y)
                print("Epoch", epoch, "- Training cost:", costs)
                
            for i in range(0, X.shape[0], batch_size):
                end_point = min(i + batch_size, X.shape[0])
                self.update(X[i:end_point,:], Y[i:end_point], eta)
                
        if verbose:
            predicted_y = self.predict(X)
            costs = self.cost(Y, predicted_y)
            print("Epoch", epochs, "- Training cost:", costs)

            
        
    def update(self, x, y, eta):
        """Updates neural network weights based on new training data.
        
        Args:
          - x: an array of length `input_units`
          - y: a float representing the output
        """
        prediction = self.predict(x)
        dldh = self.cost_derivative(y, prediction)
        for i, layer in enumerate(self.layers[::-1]):
            dldh = layer.back_prop(dldh, eta)
            
