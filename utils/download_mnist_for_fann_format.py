import tensorflow as tf
import numpy as np

# download MNIST  from tensorflow
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize
x_train, x_test = x_train / 255.0, x_test / 255.0

# flatten the input 
x_train_flat = x_train.reshape(-1, 784)
x_test_flat = x_test.reshape(-1, 784)

# change the labels to fann format
y_train_one_hot = np.zeros((y_train.size, 10))
y_train_one_hot[np.arange(y_train.size), y_train] = 1

y_test_one_hot = np.zeros((y_test.size, 10))
y_test_one_hot[np.arange(y_test.size), y_test] = 1

# function change to FANN dataset format
def prepare_fann_data(inputs, outputs):
    num_samples = inputs.shape[0]
    num_inputs = inputs.shape[1]
    num_outputs = outputs.shape[1]

    fann_data = []
    fann_data.append(f"{num_samples} {num_inputs} {num_outputs}")

    for input_row, output_row in zip(inputs, outputs):
        input_str = ' '.join(map(str, input_row))
        output_str = ' '.join(map(str, output_row))
        fann_data.append(f"{input_str}\n{output_str}")

    return '\n'.join(fann_data)

# save training data
train_fann_data = prepare_fann_data(x_train_flat, y_train_one_hot)
train_file_path = 'mnist_train_fann.data'
with open(train_file_path, 'w') as f:
    f.write(train_fann_data)

# save testing data
test_fann_data = prepare_fann_data(x_test_flat, y_test_one_hot)
test_file_path = 'mnist_test_fann.data'
with open(test_file_path, 'w') as f:
    f.write(test_fann_data)

print(f"FANN formatted training data saved to {train_file_path}")
print(f"FANN formatted testing data saved to {test_file_path}")