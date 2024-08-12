#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <fann.h>

#include "Federated_Learning.c"
#include "Federated_Learning_parallel_clients.h"


void* train_client_thread(void* args) {
    struct training_thread_args* targs = (struct training_thread_args*)args;
    printf("starting training on client %d \n", targs->client_id);
    //anns[i] = train_client_preloaded(data[i], global_model, clients_selected_for_training[i], epochs_of_training); 
    targs->anns[targs->client_index] = train_client_preloaded(targs->data[targs->client_index], targs->global_model, targs->client_id, targs->epochs_of_training);
    pthread_exit(NULL);
}
/*void run_training_on_clients_preloaded_data(struct fann** anns, struct fann_train_data **data, struct fann *global_model, int num_participating_devices, int *clients_selected_for_training, int epochs_of_training){

    for(int i = 0; i < num_participating_devices; i++){
        printf("starting training on client %d \n", clients_selected_for_training[i]);
        anns[i] = train_client_preloaded(data[i], global_model, clients_selected_for_training[i], epochs_of_training); 
    }
}
*/

void run_training_on_clients_preloaded_data_parallel(struct fann** anns, struct fann_train_data **data, struct fann *global_model, int num_participating_devices, int *clients_selected_for_training, int epochs_of_training) {
    pthread_t threads[num_participating_devices];
    struct training_thread_args targs[num_participating_devices];

    for(int i = 0; i < num_participating_devices; i++) {
        targs[i].anns = anns;
        targs[i].data = data;
        targs[i].global_model = global_model;
        targs[i].client_index = i;
        targs[i].client_id = clients_selected_for_training[i];
        targs[i].epochs_of_training = epochs_of_training;

        int rc = pthread_create(&threads[i], NULL, train_client_thread, (void*)&targs[i]);
        if (rc) {
            printf("Error:unable to create thread, %d\n", rc);
            exit(-1);
        }
    }

    // Join threads
    for(int i = 0; i < num_participating_devices; i++) {
        pthread_join(threads[i], NULL);
    }
}

struct fann* run_federated_learning_local_preloaded_parallel(int rounds_of_training, int epochs_of_training, int num_participating_devices, struct fann *global_model){
    struct fann_train_data *data[NUM_CLIENTS];
    int clients_selected_for_training[num_participating_devices];

    char filename[64];
    for(int id  = 0 ; id < NUM_CLIENTS; id++){

        sprintf(filename,"./data/train/emnist_train_client_%d.data", id);
        data[id] = fann_read_train_from_file(filename);
    }

    for(int rounds = 1; rounds <= rounds_of_training; rounds++){

        select_clients(num_participating_devices, clients_selected_for_training);

        struct fann* ann_list[num_participating_devices];
        run_training_on_clients_preloaded_data_parallel(ann_list, data, global_model, num_participating_devices, clients_selected_for_training, epochs_of_training);

        global_model = fed_avg_preloaded(ann_list, num_participating_devices);
    }
    return global_model;
}