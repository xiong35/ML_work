
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

nn_architecture = [
    {"input_dim": 2, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 8, "activation": "relu"},
    {"input_dim": 8, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
]


class NeuronNet:

    def gen_data(self):
        data_num = 1000
        a_x = np.linspace(0, np.pi, data_num)
        b_x = np.linspace(np.pi*0.5, np.pi*1.5, data_num)-0.2
        a_y = np.sin(a_x)+np.random.normal(-0.2, 0.2, data_num)-0.1
        b_y = np.cos(b_x)+np.random.normal(-0.2, 0.2, data_num)+0.1

        data_set = np.ones((2*data_num, 3))

        data_set[:data_num, 0] = a_x.T
        data_set[data_num:2*data_num, 0] = b_x.T

        data_set[:data_num, 1] = a_y.T
        data_set[data_num:2*data_num, 1] = b_y.T

        data_set[data_num:data_num*2, 2] = 0

        np.random.shuffle(data_set)

        train_num = int(data_num*0.8)
        train_data = data_set[:train_num, 0:2]
        train_label = data_set[:train_num, 2]
        test_data = data_set[train_num:, 0:2]
        test_label = data_set[train_num:, 2]
        plt.scatter(train_data[:,0],train_data[:,1],c=train_label,alpha=0.6,cmap='winter')
        plt.show()

        return train_data, train_label, test_data, test_label

    def init_layers(self):
        params_values = {}

        for index, layer in enumerate(nn_architecture):
            layer_index = index + 1
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            params_values['W' + str(layer_index)] = np.random.randn(
                layer_output_size, layer_input_size) * 0.5
            params_values['b' + str(layer_index)] = np.random.randn(
                layer_output_size, 1) * 0.5

        return params_values

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def relu(self, Z):
        return np.maximum(0, Z)

    def sigmoid_backward(self, dA, Z):
        sig = self.sigmoid(Z)
        return dA * sig * (1 - sig)

    def relu_backward(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def layer_forward_propagation(self, A_prev, W_curr, b_curr, activation="relu"):
        Z_curr = np.dot(W_curr, A_prev) + b_curr

        if activation == "relu":
            activation_func = self.relu
        else:
            activation_func = self.sigmoid

        return activation_func(Z_curr), Z_curr

    def forward_propagation(self, X, params_values):
        memory = {}
        A_curr = X

        for index, layer in enumerate(nn_architecture):
            layer_index = index + 1
            A_prev = A_curr

            activ_function_curr = layer["activation"]
            W_curr = params_values["W" + str(layer_index)]
            b_curr = params_values["b" + str(layer_index)]
            A_curr, Z_curr = self.layer_forward_propagation(
                A_prev, W_curr, b_curr, activ_function_curr)

            memory["A" + str(index)] = A_prev
            memory["Z" + str(layer_index)] = Z_curr

        return A_curr, memory

    def get_cost_value(self, Y_hat, Y):
        m = Y_hat.shape[1]
        cost = -1 / m * (np.dot(np.log(Y_hat), Y) +
                         np.dot(np.log(1 - Y_hat), 1 - Y))
        return np.squeeze(cost)

    def convert_prob_into_class(self, probs):
        probs_ = np.copy(probs)
        probs_[probs_ > 0.5] = 1
        probs_[probs_ <= 0.5] = 0
        return probs_

    def get_accuracy_value(self, Y_hat, Y):
        Y_hat_ = self.convert_prob_into_class(Y_hat)
        return (Y_hat_ == Y.T).mean()

    def layer_backward_propagation(self, dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
        m = A_prev.shape[1]

        if activation is "relu":
            backward_activation_func = self.relu_backward
        else:
            backward_activation_func = self.sigmoid_backward

        dZ_curr = backward_activation_func(dA_curr, Z_curr)
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

    def backward_propagation(self, Y_hat, Y, memory, params_values):
        grads_values = {}
        Y = Y.reshape(Y_hat.shape)

        dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

        for layer_index_prev, layer in reversed(list(enumerate(nn_architecture))):
            layer_index_curr = layer_index_prev + 1
            activ_function_curr = layer["activation"]

            dA_curr = dA_prev

            A_prev = memory["A" + str(layer_index_prev)]
            Z_curr = memory["Z" + str(layer_index_curr)]
            W_curr = params_values["W" + str(layer_index_curr)]
            b_curr = params_values["b" + str(layer_index_curr)]

            dA_prev, dW_curr, db_curr = self.layer_backward_propagation(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

            grads_values["dW" + str(layer_index_curr)] = dW_curr
            grads_values["db" + str(layer_index_curr)] = db_curr

        return grads_values

    def update(self, params_values, grads_values, learning_rate):
        for layer_index in range(1, len(nn_architecture)+1):
            params_values["W" + str(layer_index)] -= learning_rate * \
                grads_values["dW" + str(layer_index)]
            params_values["b" + str(layer_index)] -= learning_rate * \
                grads_values["db" + str(layer_index)]

        return params_values

    def train(self, X, Y, epochs=10, learning_rate=1e-2):
        params_values = self.init_layers()
        history = dict()
        cost_history = []
        accuracy_history = []

        for _ in range(epochs):
            Y_hat, memory = self.forward_propagation(
                X, params_values)
            cost = self.get_cost_value(Y_hat, Y)
            cost_history.append(cost)
            accuracy = self.get_accuracy_value(Y_hat, Y)
            accuracy_history.append(accuracy)

            grads_values = self.backward_propagation(
                Y_hat, Y, memory, params_values)
            params_values = self.update(params_values, grads_values, learning_rate)

        history['acc'] = accuracy_history
        history['loss'] = cost_history

        return params_values, history


nn = NeuronNet()
X, Y, x_test, y_test = nn.gen_data()
params_values, history = nn.train(
    X.T, Y.reshape(1, -1).T, epochs=100, learning_rate=0.03)
acc = history['acc']
loss = history['loss']
epochs = range(1, len(acc)+1)
plt.plot(epochs, loss, 'bo', alpha=0.7, label='Training loss')
plt.legend()
plt.show()
plt.clf()
plt.plot(epochs, acc, label='Training acc')
plt.legend()
plt.show()
