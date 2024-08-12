import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import os


NUM_clients = 100 


def preprocess_emnist(client_data, client_id):
    client_dataset = client_data.create_tf_dataset_for_client(client_id)
    X, y = [], []
    
    for example in client_dataset:
        image = tf.reshape(example['pixels'], [-1]).numpy() / 255.0  # flatten the input and normalize all parameters
        label = np.zeros(10)                                         # shaping the output layer 
        label[example['label'].numpy()] = 1
        X.append(image)
        y.append(label)
    
    return np.array(X), np.array(y)

def save_to_fann_format(x, y, file_name, append=False):
    mode = 'a' if append else 'w'
    num_samples, num_inputs = x.shape
    num_outputs = y.shape[1]
    
    with open(file_name, mode) as f:
        if not append:
            f.write(f"{num_samples} {num_inputs} {num_outputs}\n")
        for i in range(num_samples):
            inputs = " ".join(map(str, x[i]))
            outputs = " ".join(map(str, y[i]))
            f.write(f"{inputs}\n{outputs}\n")

# download EMNIST dataset from tff
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()



os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)

client_ids = emnist_train.client_ids[:NUM_clients]

# making the training data for the clients
i = 0
for client_id in client_ids:
    X_train, y_train = preprocess_emnist(emnist_train, client_id)
    train_file_name = f'train/emnist_train_client_{i}.data'
    save_to_fann_format(X_train, y_train, train_file_name)
    print(f"training data {client_id} saved in FANN format as {train_file_name}")
    i += 1

# save test data from all clients in one file
X_test_all, y_test_all = [], []
for client_id in client_ids:
    X_test, y_test = preprocess_emnist(emnist_test, client_id)
    X_test_all.extend(X_test)
    y_test_all.extend(y_test)

X_test_all = np.array(X_test_all)
y_test_all = np.array(y_test_all)
test_file_name = 'test/emnist_test_all.data'
save_to_fann_format(X_test_all, y_test_all, test_file_name)
print(f"Test data saved in FANN format as {test_file_name}")