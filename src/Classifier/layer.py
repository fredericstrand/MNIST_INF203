class Layer:
    def __init__(self, input_size, output_size):
        # Initialize weights with small random values and biases to 0
        self._weights = np.random.randn(input_size, output_size) * 0.1
        self._bias = np.zeros((1, output_size))
        self.adam_weights = AdamOptimizer(learning_rate)
        self.adam_bias = AdamOptimizer(learning_rate)

    def forward(self, inputs):
        # Calculate the weighted sum of inputs + biases
        self._inputs = inputs
        self.output = np.dot(inputs, self._weights) + self._bias

    def backward(self, d_output, learning_rate):
        # Gradient of weights and biases based on the error signal (d_output)
        d_weights = np.dot(self._inputs.T, d_output)
        d_bias = np.sum(d_output, axis=0, keepdims=True)

        # Gradient of the input to pass to the previous layer
        d_inputs = np.dot(d_output, self._weights.T)

        # Update weights and biases
        self._weights = self.adam_weights.update(self._weights, d_weights)
        self._bias = self.adam_bias.update(self._bias, d_bias)

        return d_inputs