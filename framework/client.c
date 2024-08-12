#include "sockets/FL_client.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <fann.h>

#define TIME_SLEEP 120
#define EPOCHS 5

int main(int args, char* argv[]){
    char* ip_addr;
    strcpy(ip_addr, argv[1]);

    struct fann *ann;
    int need_training = 1;
    char last_trained_version[64];
    strcpy(last_trained_version, "NULL");
    char current_version[64];
    struct fann_train_data *train_data;

    char *filename = "./data/local.data";

    while (need_training)
    {
        ann = connect_and_get_network(&need_training, current_version, "127.0.0.1");

        if(strcmp(last_trained_version, current_version) == 0){
            printf("already have up to date network\n");
            sleep(TIME_SLEEP);
            continue;
        }
        printf("training network...\n");
        fann_train_on_file(ann, filename, EPOCHS, 100, 0.0);

        connect_and_send_network(ann, "127.0.0.1");

        printf("sent trained network back to the server\n");
        strcpy(last_trained_version, current_version);

        sleep(TIME_SLEEP);
    }
}