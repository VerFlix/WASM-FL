#ifndef __FL_parallel_clients__
#define __FL_parallel_clients__

struct training_thread_args {
    struct fann** anns;
    struct fann_train_data** data;
    struct fann* global_model;
    int client_index;
    int client_id;
    int epochs_of_training;
};

void* train_client_thread(void* args);


void run_training_on_clients_preloaded_data_parallel(struct fann** anns, struct fann_train_data **data, struct fann *global_model, int num_participating_devices, int *clients_selected_for_training, int epochs_of_training);

struct fann* run_federated_learning_local_preloaded_parallel(int rounds_of_training, int epochs_of_training, int num_participating_devices,  struct fann *global_model);

#endif //__FL_parallel_clients__