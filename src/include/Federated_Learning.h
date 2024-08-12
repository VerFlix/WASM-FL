#ifndef __fed_learning__
#define __fed_learning__

#include <fann_data.h>

struct fann* initialize();

struct fann* load_existing_network(char* filename);

void fed_avg_from_files(int num_clients, int* selected_clients);

struct fann* fed_avg_preloaded(struct fann **ann_list, int num_clients);

void select_clients(int num_clients, int* selected_clients, int max_client);

void train_client_fileIO(int client_ID, int epochs);

void set_weight_array(struct fann *ann, struct fann_connection *connections, unsigned int num_connections);

struct fann* train_client_preloaded(struct fann_train_data* data, struct fann *ann, int client_ID, int epochs_of_training);

void run_training_on_clients_fileIO(int num_participating_devices, int *clients_selected_for_training, int epochs_of_training);

void run_training_on_clients_preloaded_data(struct fann** anns, struct fann_train_data **data, struct fann *global_model, int num_participating_devices, int *clients_selected_for_training, int epochs_of_training);

int run_federated_learning_local_fileIO(int rounds_of_training, int epochs_of_training, int num_participating_devices);

struct fann* run_federated_learning_local_preloaded(int rounds_of_training, int epochs_of_training, int num_participating_devices, struct fann *global_model);



#endif // __fed_learning__