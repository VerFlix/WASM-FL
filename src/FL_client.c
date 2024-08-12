#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <fann.h>
#include "FL_client.h"

#define PORT 8088
#define BUFFER_SIZE 32768  

void send_instructiondata(int socket, char *instructiondata){
    char instructiondata_send[64];
    snprintf(instructiondata_send, sizeof(instructiondata_send), "%s\n", instructiondata);
    if (send(socket, instructiondata_send, strlen(instructiondata_send), 0) < 0) {
        perror("send failed");
        exit(EXIT_FAILURE);
    }
}

void send_fann_network(int socket, struct fann *ann) {
    char *buffer;
    size_t size;
    //open memory stream - this way we dont have to save the ann as a file
    FILE *mem_stream = open_memstream(&buffer, &size);
    if (!mem_stream) {
        perror("open_memstream failed");
        exit(EXIT_FAILURE);
    }


    fann_save_internal_fd(ann, mem_stream,"buffer", 0);
    long data_size = ftell(mem_stream);
    fclose(mem_stream);
    
    //fann_save(ann, "temp_fann_client_side.net");

    // Send instruction data first
    send_instructiondata(socket, "POST trained_model");

    sleep(1);
    // Send the data in chunks
    size_t sent = 0;
    while (sent < size) {
        size_t to_send = (size - sent) > BUFFER_SIZE ? BUFFER_SIZE : (size - sent);
        ssize_t bytes_sent = send(socket, buffer + sent, to_send, 0);
        if (bytes_sent < 0) {
            perror("send failed");
            free(buffer);
            exit(EXIT_FAILURE);
        }
        sent += bytes_sent;
    }

    printf("Sent FANN network of size %ld\n", data_size);
}

struct fann* recieve_fann_network(int socket, int *training_needed, char* version_ID){
    char *mem_buffer, rec_buffer[BUFFER_SIZE];
    size_t size;
    //open memory stream - this way we dont have to save the ann as a file
    FILE *mem_stream = open_memstream(&mem_buffer, &size);
    int bytes_received;

    // Receive metadata
    char metadata[64];
    bytes_received = recv(socket, metadata, sizeof(metadata), 0);
    if (bytes_received < 0) {
        perror("recv failed");
        fclose(mem_stream);
        exit(EXIT_FAILURE);
    }
    metadata[bytes_received] = '\0';
    printf("Received metadata: %s\n", metadata);

    // Process metadata
    char *training_needed_str = strtok(metadata, "\n");
    strcpy (version_ID, strtok(NULL, "\n"));
    
    if(strcmp(training_needed_str, "need_training") == 0) *training_needed = 1;
    else training_needed = 0;


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

    return fann_create_from_fd(read_mem_stream,"buffer");

    fclose(mem_stream);
}

int connect_to_server(char* ip_addr){
    int sock = 0;
    struct sockaddr_in serv_addr;

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("\n Socket creation error \n");
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    if (inet_pton(AF_INET, ip_addr, &serv_addr.sin_addr) <= 0) {
        printf("\nInvalid address/ Address not supported \n");
        return -1;
    }

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        printf("\nConnection Failed \n");
        return -1;
    }

    printf("Connected to server\n");
    return sock;
}

void connect_and_send_network(struct fann *ann, char* ip_addr){
    
    int sock = connect_to_server(ip_addr);

    send_fann_network(sock, ann);

    fann_destroy(ann);
    close(sock);
}

struct fann* connect_and_get_network(int *training_needed, char *version_ID, char* ip_addr){

    int sock = connect_to_server(ip_addr);

    send_instructiondata(sock, "GET global_model");

    return recieve_fann_network(sock, training_needed, version_ID);

    close(sock);
}
