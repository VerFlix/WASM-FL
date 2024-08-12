#ifndef __fl_server__
#define __fl_server__

void send_global_model(int socket, int training_needed, char *version_ID);

int receive_and_handle(int socket, struct fann *ann, int training_needed, char* version_ID);

void initialize_server(int *server_fd, struct sockaddr_in *address);

int parseInt(char* str);

#endif // __fl_server__