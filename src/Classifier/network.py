from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize two layers: one hidden and one output
        self.layer1 = Layer(input_size, hidden_size)
        self.layer2 = Layer(hidden_size, output_size)

    def forward(self, X):
        # Forward pass through the first layer with ReLU activation
        self.layer1.forward(X)
        self.layer1_output = relu(self.layer1.output)

        # Forward pass through the second layer with Softmax activation
        self.layer2.forward(self.layer1_output)
        self.output = softmax(self.layer2.output)

    def backward(self, X, y, learning_rate):
        # Backward pass using Cross-Entropy Loss
        loss = cross_entropy_loss(y, self.output)
        d_output = self.output - y

        # Backward pass through second layer
        d_hidden = self.layer2.backward(d_output, learning_rate)

        # Backward pass through the first layer with ReLU activation
        d_inputs = self.layer1.backward(d_hidden * relu_derivative(self.layer1_output), learning_rate)

    def train(self, X_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X_train)
            self.backward(X_train, y_train, learning_rate)

            # Print the loss every 100 epochs
            if epoch % 100 == 0:
                loss = cross_entropy_loss(y_train, self.output)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        # Forward pass for predictions
        self.forward(X)
        return np.argmax(self.output, axis=1)