#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <fann.h>
#include "FL_server.h"

#define PORT 8088
#define BUFFER_SIZE 32768  

void send_global_model(int socket, int training_needed, char *version_ID){
    // Open global_model.net
    FILE *file = fopen("./networks/global_model.net", "r");

    // Send the file in chunks
    char buffer[BUFFER_SIZE];
    size_t read_size;

    // Send metadata first
    char metadata[64];
    snprintf(metadata, sizeof(metadata), "%s\n%s\n", training_needed ? "need_training" : "not_need_training", version_ID);
    if (send(socket, metadata, strlen(metadata), 0) < 0) {
        perror("send failed");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    sleep(1);

    while ((read_size = fread(buffer, 1, BUFFER_SIZE, file)) > 0) {
        size_t sent = 0;
        while (sent < read_size) {
            ssize_t bytes_sent = send(socket, buffer + sent, read_size - sent, 0);
            if (bytes_sent < 0) {
                perror("send failed");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            sent += bytes_sent;
        }
    }
}

int receive_and_handle(int socket, struct fann *ann, int training_needed, char* version_ID) {
    char *mem_buffer, rec_buffer[BUFFER_SIZE];
    size_t size;

    //open memory stream - this way we dont have to save the ann as a file
    FILE *mem_stream = open_memstream(&mem_buffer, &size);
    int bytes_received;

    // Receive instructiondata
    char instructiondata[64];
    bytes_received = recv(socket, instructiondata, sizeof(instructiondata) - 1, 0);
    if (bytes_received < 0) {
        perror("recv failed");
        fclose(mem_stream);
        exit(EXIT_FAILURE);
    }
    instructiondata[bytes_received] = '\0';
    printf("Received instructiondata: %s\n", instructiondata);

    // Process instructions
    char *instruction = strtok(instructiondata, "\n");

    // Check if the client sends a trained model
    if(strcmp(instruction, "POST trained_model") == 0){

        while ((bytes_received = recv(socket, rec_buffer, BUFFER_SIZE, 0)) > 0) {
        size_t bytes_written = fwrite(rec_buffer, 1, bytes_received, mem_stream);

        if (bytes_written != bytes_received) {
            perror("fwrite failed");
            fclose(mem_stream);
            exit(EXIT_FAILURE);
        }
    }
    fclose(mem_stream);

    FILE *read_mem_stream = fmemopen(mem_buffer, size, "r");

    ann = fann_create_from_fd(read_mem_stream,"buffer");
    printf("Received and created FANN network\n");
    return 1;

    }
    // Check if the client requests the global_model
    else if (strcmp(instruction, "GET global_model") == 0){
        send_global_model(socket, training_needed, version_ID);
        return 0;
    }

}

void initialize_server(int *server_fd, struct sockaddr_in *address){

    int opt = 1;

    if ((*server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    if (setsockopt(*server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    address->sin_family = AF_INET;
    address->sin_addr.s_addr = INADDR_ANY;
    address->sin_port = htons(PORT);

    if (bind(*server_fd, (struct sockaddr *)address, sizeof(*address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    if (listen(*server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }
}


int parseInt(char* str) {
    int res = 0;
    for (int i = 0; str[i] != '\0'; ++i) {
        res = res * 10 + str[i] - '0';
    }
    return res;
}