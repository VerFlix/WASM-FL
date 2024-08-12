#ifndef __fl_client__
#define __fl_client__

#include <fann_data.h>

void send_instructiondata(int socket, char *instructiondata);

void send_fann_network(int socket, struct fann *ann);

struct fann* recieve_fann_network(int socket, int *training_needed, char* version_ID);

int connect_to_server(char* ip_addr);

void connect_and_send_network(struct fann *ann, char* ip_addr);

struct fann* connect_and_get_network(int *training_needed, char *version_ID, char* ip_addr);

#endif // __fl_client__