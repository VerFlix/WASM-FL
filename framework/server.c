#include "sockets/FL_server.c"
#include "Federated_Learning.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <fann.h>

int main(int args, char* argv[]) {
    int num_clients = parseInt(argv[1]);
    struct fann *ann, *ann_list[num_clients];

    int new_ann;
    int rounds_of_training = 5;

    struct sockaddr_in address;
    int server_fd, new_socket;
    int addrlen = sizeof(address);

    initialize_server(&server_fd, &address);
    int clients_needed = num_clients;

    printf("Server listening on port %d\n", PORT);

    while (rounds_of_training) {
        //TODO: new thread for every connection

        char current_version[32];
        sprintf(current_version, "%d rounds left\n", rounds_of_training);

        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
            perror("accept");
            exit(EXIT_FAILURE);
        }

        printf("Accepted connection from client\n");
        new_ann = receive_and_handle(new_socket, ann, 1, current_version);
        
        if (new_ann)
        {
            printf("got newly trained ann: %d trained networks left to start fed_avg\n", clients_needed - 1);
            ann_list[num_clients - clients_needed] = ann;
            clients_needed --;
        }
        if (clients_needed == 0)
        {
            clients_needed = num_clients;
            ann = fed_avg_preloaded(ann_list, num_clients);
            fann_save(ann, "./networks/global_model.net");

            rounds_of_training --;
            sprintf(current_version, "%d rounds left\n", rounds_of_training);
        }
        
        
        close(new_socket);
    }

    close(server_fd);
    return 0;
}