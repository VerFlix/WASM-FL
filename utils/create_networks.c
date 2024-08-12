#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "fann.h"

#define NEURONS_INPUT_LAYER 784
#define NEURONS_OUTPUT_LAYER 10

int main(){
    struct fann *ann;
    int neurons_input_layer;
    char save_name[80];
    for(int i = 1; i <= 8 ; i= i+i){ // number of hidden layers
        for(int j = 64; j<= 256 ; j = j+j){ // number of neurons per layer
            switch (i)
            {
            case 1:
                ann = fann_create_standard(i + 2, NEURONS_INPUT_LAYER, j, NEURONS_OUTPUT_LAYER);
                break;
            
            case 2:
                ann = fann_create_standard(i+2, NEURONS_INPUT_LAYER, j, j , NEURONS_OUTPUT_LAYER);
                break;

            case 4:
                ann = fann_create_standard(i+2, NEURONS_INPUT_LAYER, j , j , j , j, NEURONS_OUTPUT_LAYER);
                break;

            case 8:
                ann = fann_create_standard(i+2, NEURONS_INPUT_LAYER, j, j, j, j, j, j, j, j, NEURONS_OUTPUT_LAYER);
                break;
            }

            fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
            fann_set_activation_function_output(ann, FANN_SIGMOID);

            *save_name = '\0';
            sprintf(save_name,"./network_%d-%d.net", i,j);
            fann_save(ann, save_name);
        }
    }
}