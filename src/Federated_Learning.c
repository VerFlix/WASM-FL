#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "fann.h"
#include "Federated_Learning.h"


struct fann* initialize(){
    const unsigned int num_layers = 6;
	const unsigned int num_neurons_hidden = 256;
    struct fann *ann;
    ann = fann_create_standard(num_layers,
					  784, num_neurons_hidden, num_neurons_hidden, num_neurons_hidden, num_neurons_hidden, 10);

    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID);

    fann_save(ann, "./networks/global_model.net");
    return ann;
}

struct fann* load_existing_network(char* filename){
    struct fann *ann = fann_create_from_file(filename);

    fann_save(ann, "./networks/global_model.net");
    return ann;
}

void fed_avg_from_files(int num_clients, int* selected_clients){
    char i_str[16];
    char str[80];
    struct fann *client;

    printf("loading first Network\n");

    *str = '\0';
    sprintf(i_str,"%d.net", selected_clients[0]);
    client = fann_create_from_file(strcat (strcat(str , "./networks/emnist_float_"), i_str));

    printf("Network loaded ... performing FedAvg\n");
    unsigned int connection_num = fann_get_total_connections(client);
    struct fann_connection *connections, *connections_average;
    connections_average = (struct fann_connection*)calloc(connection_num, sizeof(struct fann_connection));
    connections = (struct fann_connection*)calloc(connection_num, sizeof(struct fann_connection));
    fann_get_connection_array(client,connections_average);

    for(int j = 1; j <= num_clients - 1; j++){
        *str = '\0';
        sprintf(i_str,"%d.net", selected_clients[j]);
        client = fann_create_from_file(strcat (strcat(str , "./networks/emnist_float_"), i_str));
        fann_get_connection_array(client,connections);
        for(int i = 0; i< connection_num; ++i){
            if(connections_average[i].from_neuron != connections[i].from_neuron || connections_average[i].to_neuron != connections[i].to_neuron) printf("Connection Arrays not matching!");
            connections_average[i].weight += connections[i].weight;
        }
    }
    free(connections);
    for(int i = 0; i< connection_num; ++i){
        connections_average[i].weight = connections_average[i].weight/num_clients;
        //printf("weight from %u to %u: %f\n", connections[i].from_neuron, connections[i].to_neuron, connections[i].weight);
    }
    fann_set_weight_array(client, connections_average, connection_num);
    fann_save(client, "./networks/global_model.net");
    free(connections_average);
}

void set_weight_array(struct fann *ann, struct fann_connection *connections, unsigned int num_connections){
    
    /*This function is deeply inspired by the fann_get_connection_array function as it is esentially in inverse of that.*/

    //printf("weight from %u to %u: %f\n", connections[i].from_neuron, connections[i].to_neuron, connections[i].weight);    
    int i = 0;
    struct fann_neuron *first_neuron;
    struct fann_layer *layer_it;
    struct fann_neuron *neuron_it;
    unsigned int idx;
    unsigned int source_index;
    unsigned int destination_index;

    first_neuron = ann->first_layer->first_neuron;

    source_index = 0;
    destination_index = 0;

    /* The following assumes that the last unused bias has no connections */

    /* for each layer */
    for (layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++) {
        /* for each neuron */
        for (neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++) {
        /* for each connection */
            for (idx = neuron_it->first_con; idx < neuron_it->last_con; idx++) {
                /* Assign the source, destination and weight */

                ann->weights[source_index] = connections->weight;

                connections++;
                source_index++;
            }
        destination_index++;
        }
    }
}


struct fann* fed_avg_preloaded(struct fann **ann_list, int num_clients){



    printf("Networks trained ... performing FedAvg\n");

    unsigned int connection_num = fann_get_total_connections(ann_list[0]);
    struct fann_connection *connections, *connections_average;
    connections_average = (struct fann_connection*)calloc(connection_num, sizeof(struct fann_connection));
    connections = (struct fann_connection*)calloc(connection_num, sizeof(struct fann_connection));
    fann_get_connection_array(ann_list[0], connections_average);


    for(int j = 1; j <= num_clients - 1; j++){

        fann_get_connection_array(ann_list[j] ,connections);
        for(int i = 0; i< connection_num; ++i){
            if(connections_average[i].from_neuron != connections[i].from_neuron || connections_average[i].to_neuron != connections[i].to_neuron) printf("Connection Arrays not matching!");
            connections_average[i].weight += connections[i].weight;
            //printf("weight from %u to %u: %f\n", connections[i].from_neuron, connections[i].to_neuron, connections[i].weight);
        }
    }

    for(int i = 0; i< connection_num; ++i){
        connections_average[i].weight = connections_average[i].weight/num_clients;
    }

    free(connections);

    set_weight_array(ann_list[0], connections_average, connection_num);
    free(connections_average);

    printf("FedAvg done\n");

    return ann_list[0];
}

void select_clients(int num_clients, int* selected_clients, int max_client){
    // Generate a number from min to max
    unsigned int min = 0;
    unsigned int max = max_client -1;

    if (num_clients > max)
    {
        printf("Amount must be less than max.\n");
        return;
    }

    // Seed srand
    srand(time(0));

    int i = 0;
    // Generate random numbers until we've reached the desired amount
    while(i < num_clients)
    {
        // Generate a random number within the range [min, max]
        int num = rand() % max + min;

        // If the number doesn't exist yet, add it to the vector
        int found = 0;
        for (int j = 0; j < i ; j++){
            if (num == selected_clients[j]){
                found++;
            }
        }
        if (!found){
            selected_clients[i] = num;
            printf("num = %d\n", num);
            i++;
        }
    }
}

void train_client_fileIO(int client_ID, int epochs){

    struct fann *ann = fann_create_from_file("./networks/global_model.net");

    char filename[64];
    sprintf(filename,"./data/train/emnist_train_client_%d.data", client_ID);

    struct fann_train_data *training_dataset = fann_read_train_from_file(filename);
    fann_train_on_data(ann, training_dataset, epochs, 100, 0.0);

    sprintf(filename,"./networks/emnist_float_%d.net", client_ID);

    fann_save(ann, filename);
}

struct fann* train_client_preloaded(struct fann_train_data* data, struct fann *ann, int client_ID, int epochs_of_training){
    fann_train_on_data(ann, data, epochs_of_training, 100, 0.0);
    return ann;
}

void run_training_on_clients_fileIO(int num_participating_devices, int *clients_selected_for_training, int epochs_of_training){

    //TODO: initialize a thread for every client for parallel mode
    for(int i = 0; i < num_participating_devices; i++){
        printf("starting training on client %d \n", clients_selected_for_training[i]);
        train_client_fileIO(clients_selected_for_training[i], epochs_of_training); 

    }

}

void run_training_on_clients_preloaded_data(struct fann** anns, struct fann_train_data **data, struct fann *global_model, int num_participating_devices, int *clients_selected_for_training, int epochs_of_training){

    for(int i = 0; i < num_participating_devices; i++){
        printf("starting training on client %d \n", clients_selected_for_training[i]);
        anns[i] = train_client_preloaded(data[i], global_model, clients_selected_for_training[i], epochs_of_training); 
    }
}

int run_federated_learning_local_fileIO(int rounds_of_training, int epochs_of_training, int num_participating_devices){
    initialize();
    int clients_selected_for_training[num_participating_devices];

    for(int rounds = 1; rounds <= rounds_of_training; rounds++){

        select_clients(num_participating_devices, clients_selected_for_training, num_participating_devices);

        run_training_on_clients_fileIO(num_participating_devices, clients_selected_for_training, epochs_of_training);

        fed_avg_from_files(num_participating_devices, clients_selected_for_training);
    }
    return 0;
}

struct fann* run_federated_learning_local_preloaded(int rounds_of_training, int epochs_of_training, int num_participating_devices, struct fann *global_model){
    struct fann_train_data *data[num_participating_devices];
    int clients_selected_for_training[num_participating_devices];

    for(int i = 0; i < num_participating_devices;i++){
        clients_selected_for_training[i] = i;
    }

    char filename[64];
    for(int id  = 0 ; id < num_participating_devices; id++){

        sprintf(filename,"./data/emnist_train_client_%d.data", id);
        data[id] = fann_read_train_from_file(filename);
    }

    for(int rounds = 1; rounds <= rounds_of_training; rounds++){

        //select_clients(num_participating_devices, clients_selected_for_training, num_participating_devices);

        struct fann* ann_list[num_participating_devices];
        run_training_on_clients_preloaded_data(ann_list, data, global_model, num_participating_devices, clients_selected_for_training, epochs_of_training);

        global_model = fed_avg_preloaded(ann_list, num_participating_devices);
    }
    return 0;
}

//int main(){
//
//    run_federated_learning_local_preloaded(5,2,12);
//
//}