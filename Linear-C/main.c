#include "include/backprop.h"
#include "include/layer.h"
#include "include/neuron.h"
#include "include/read_csv.h"

layer *lay = NULL;
int num_layers;
int *num_neurons;
float alpha;
float *cost;
float full_cost;
float **input;
float **desired_outputs;
int num_training_ex;
int n=1;

int main(void)
{
    int i;

    srand(time(0));

    num_layers = 4; //网络层数

    num_neurons = (int*) malloc(num_layers * sizeof(int));
    memset(num_neurons,0,num_layers *sizeof(int));

    //各层的神经元个数
    num_neurons[0]=4;
    num_neurons[1]=8;
    num_neurons[2]=4;
    num_neurons[3]=1;

    printf("\n");

    // Initialize the neural network module
    if(init()!= SUCCESS_INIT)
    {
        printf("Error in Initialization...\n");
        exit(0);
    }

    alpha = 0.15; //学习率

    num_training_ex = 1096; //训练样本数

    input = (float**) malloc(num_training_ex * sizeof(float*));
    for(i=0;i<num_training_ex;i++)
    {
        input[i] = (float*)malloc(num_neurons[0] * sizeof(float));
    }

    desired_outputs = (float**) malloc(num_training_ex* sizeof(float*));
    for(i=0;i<num_training_ex;i++)
    {
        desired_outputs[i] = (float*)malloc(num_neurons[num_layers-1] * sizeof(float));
    }

    cost = (float *) malloc(num_neurons[num_layers-1] * sizeof(float));
    memset(cost,0,num_neurons[num_layers-1]*sizeof(float));

    get_inputs(); //获得输入

    get_desired_outputs(); //获得标签

    train_neural_net(); //训练网络
    test_nn(); //测试网络

    if(dinit()!= SUCCESS_DINIT)
    {
        printf("Error in Dinitialization...\n");
    }

    return 0;
}



