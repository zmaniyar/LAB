import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the neural network class
class NeuralNetwork:
    def __init__(self, x, y):
        # Initialize the input and output layers
        self.input = x
        self.output = y
        # Define the number of neurons in the hidden layer
        self.hidden_size = 4
        # Initialize the weight matrices with random values
        self.weights1 = np.random.randn(self.input.shape[1], self.hidden_size)
        self.weights2 = np.random.randn(self.hidden_size, 1)

    def feedforward(self):
        # Compute the dot product of the input and the first set of weights
        self.hidden = sigmoid(np.dot(self.input, self.weights1))
        # Compute the dot product of the hidden layer and the second set of weights
        self.predicted_output = sigmoid(np.dot(self.hidden, self.weights2))

    def backpropagate(self):
        # Compute the error in the output layer
        output_error = self.output - self.predicted_output
        # Compute the derivative of the error with respect to the predicted output
        d_predicted_output = output_error * sigmoid_derivative(self.predicted_output)
        # Compute the error in the hidden layer
        hidden_error = d_predicted_output.dot(self.weights2.T)
        # Compute the derivative of the error with respect to the hidden layer
        d_hidden = hidden_error * sigmoid_derivative(self.hidden)
        # Update the weights
        self.weights1 += self.input.T.dot(d_hidden)
        self.weights2 += self.hidden.T.dot(d_predicted_output)

    def train(self, epochs):
        for epoch in range(epochs):
            # Compute the output using forward propagation
            self.feedforward()
            # Update the weights using backpropagation
            self.backpropagate()

    def predict(self, x):
        # Compute the dot product of the input and the first set of weights
        hidden = sigmoid(np.dot(x, self.weights1))
        # Compute the dot product of the hidden layer and the second set of weights
        predicted_output = sigmoid(np.dot(hidden, self.weights2))
        return predicted_output


# Define the input and output data
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create a neural network instance and train it
nn = NeuralNetwork(X, y)
nn.train(10000)

# Make predictions on new data
x_test = np.array([[0, 0, 0], [1, 0, 0]])
for x in x_test:
    print("Input:", x)
    print("Output:", nn.predict(x))
