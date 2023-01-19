//
// Created by 张智行 on 2023/1/18.
//
#include "../include/backprop.h"
#include "../include/read_csv.h"
extern layer *lay;
extern int num_layers;
extern int *num_neurons;
extern float alpha;
extern float *cost;
extern float full_cost;
extern float **input;
extern float **desired_outputs;
extern int num_training_ex;
extern int n;

typedef struct {
    int n_hidden;
    int* hidden_layers_size;
    int* hidden_activation_functions;
    double learning_rate;
    int n_iterations_max;
    int momentum;
    int output_layer_size;
    int output_activation_function;
    double** data_train;
    double** data_test;
    int feature_size;
    int train_sample_size;
    int test_sample_size;
    double*** weight;
} parameters;

parameters* read_data() //初始化参数结构体并将读取到的数据保存至结构体变量param中
{
    parameters* param = (parameters*)malloc(sizeof(parameters));

    char* train_filename = "C:\\Users\\ZZX\\CLionProjects\\data\\data\\data_train.csv";
    param->train_sample_size = 1096;
    // Feature size = Number of input features + 1 output feature
    param->feature_size = 5;

    // Create 2D array memory for the dataset
    param->data_train = (double**)malloc(param->train_sample_size * sizeof(double*));
    for (int i = 0; i < param->train_sample_size; i++)
        param->data_train[i] = (double*)malloc(param->feature_size * sizeof(double));

    // Read the train dataset from the csv into the 2D array
    read_csv(train_filename, param->train_sample_size, param->feature_size, param->data_train);
    return param;
}

int init()
{
    if(create_architecture() != SUCCESS_CREATE_ARCHITECTURE)
    {
        printf("Error in creating architecture...\n");
        return ERR_INIT;
    }

    printf("Neural Network Created Successfully...\n\n");
    return SUCCESS_INIT;
}

//Get Inputs
void  get_inputs() //获得输入数据
{
    parameters* param = read_data();
    int i,j;

    for(i=0;i<num_training_ex;i++)
    {
        printf("Enter the Inputs for training example[%d]:\n",i);

        for(j=0;j<num_neurons[0];j++)
        {
//            scanf("%f",&input[i][j]);
            input[i][j]= 0.1*param->data_train[i][j];
            printf("%f\n",input[i][j]);
        }
        printf("\n");
    }
}

//Get Labels
void get_desired_outputs() //获得数据标签
{
    parameters* param = read_data();
    int i,j;

    for(i=0;i<num_training_ex;i++)
    {
        for(j=0;j<num_neurons[num_layers-1];j++)
        {
            printf("Enter the Desired Outputs (Labels) for training example[%d]: \n",i);
            desired_outputs[i][j] = param->data_train[i][4];
            printf("%f\n",desired_outputs[i][j]);
        }
    }
}

// Feed inputs to input layer
void feed_input(int i) //将数据输入到网络中
{
    int j;

    for(j=0;j<num_neurons[0];j++)
    {
        lay[0].neu[j].actv = input[i][j];
        printf("Input: %f\n",lay[0].neu[j].actv);
    }
}

// Create Neural Network Architecture
int create_architecture() //创建网络结构（create_layer create_neuron）
{
    int i=0,j=0;
    lay = (layer*) malloc(num_layers * sizeof(layer));

    for(i=0;i<num_layers;i++)
    {
        lay[i] = create_layer(num_neurons[i]);
        lay[i].num_neu = num_neurons[i];
        printf("Created Layer: %d\n", i+1);
        printf("Number of Neurons in Layer %d: %d\n", i+1,lay[i].num_neu);

        for(j=0;j<num_neurons[i];j++)
        {
            if(i < (num_layers-1))
            {
                lay[i].neu[j] = create_neuron(num_neurons[i+1]);
            }

            printf("Neuron %d in Layer %d created\n",j+1,i+1);
        }
        printf("\n");
    }

    printf("\n");

    // Initialize the weights
    if(initialize_weights() != SUCCESS_INIT_WEIGHTS)
    {
        printf("Error Initilizing weights...\n");
        return ERR_CREATE_ARCHITECTURE;
    }

    return SUCCESS_CREATE_ARCHITECTURE;
}

int initialize_weights(void) //随机初始化权重和偏置
{
    int i,j,k;

    if(lay == NULL)
    {
        printf("No layers in Neural Network...\n");
        return ERR_INIT_WEIGHTS;
    }

    printf("Initializing weights...\n");

    for(i=0;i<num_layers-1;i++)
    {

        for(j=0;j<num_neurons[i];j++)
        {
            for(k=0;k<num_neurons[i+1];k++)
            {
                // Initialize Output Weights for each neuron
                lay[i].neu[j].out_weights[k] = ((double)rand())/((double)RAND_MAX);
                printf("%d:w[%d][%d]: %f\n",k,i,j, lay[i].neu[j].out_weights[k]);
                lay[i].neu[j].dw[k] = 0.0;
            }

            if(i>0)
            {
                lay[i].neu[j].bias = ((double)rand())/((double)RAND_MAX);
            }
        }
    }
    printf("\n");

    for (j=0; j<num_neurons[num_layers-1]; j++)
    {
        lay[num_layers-1].neu[j].bias = ((double)rand())/((double)RAND_MAX);
    }

    return SUCCESS_INIT_WEIGHTS;
}

// Train Neural Network
void train_neural_net(void) //训练网络
{
    int i;
    int it=0;

    // Gradient Descent
    for(it=0;it<1000;it++)
    {
        for(i=0;i<num_training_ex;i++)
        {
            feed_input(i); //输入数据
            forward_prop(); //前向传播
            compute_cost(i); //计算误差/损失
            back_prop(i); //反向传播
            update_weights(); //更新参数
        }
    }
}

void update_weights(void) //更新参数
{
    int i,j,k;

    for(i=0;i<num_layers-1;i++)
    {
        for(j=0;j<num_neurons[i];j++)
        {
            for(k=0;k<num_neurons[i+1];k++)
            {
                // Update Weights
                lay[i].neu[j].out_weights[k] = (lay[i].neu[j].out_weights[k]) - (alpha * lay[i].neu[j].dw[k]);
            }

            // Update Bias
            lay[i].neu[j].bias = lay[i].neu[j].bias - (alpha * lay[i].neu[j].dbias);
        }
    }
}

void forward_prop(void) //前向传播
{
    int i,j,k;

    for(i=1;i<num_layers;i++)
    {
        for(j=0;j<num_neurons[i];j++)
        {
            lay[i].neu[j].z = lay[i].neu[j].bias;

            for(k=0;k<num_neurons[i-1];k++)
            {
                lay[i].neu[j].z  = lay[i].neu[j].z + ((lay[i-1].neu[k].out_weights[j])* (lay[i-1].neu[k].actv));
            }

            //隐藏层使用Relu激活函数
            if(i < num_layers-1)
            {
                if((lay[i].neu[j].z) < 0)
                {
                    lay[i].neu[j].actv = 0;
                }

                else
                {
                    lay[i].neu[j].actv = lay[i].neu[j].z;
                }
            }

            //输出层使用sigmoid激活函数
            else
            {
                lay[i].neu[j].actv = 1/(1+exp(-lay[i].neu[j].z));
                printf("Output: %f\n", (double)(lay[i].neu[j].actv));
                printf("\n");
            }
        }
    }
}

//计算误差/损失
void compute_cost(int i)
{
    int j;
    float tmpcost=0;
    float tcost=0;

    for(j=0;j<num_neurons[num_layers-1];j++)
    {
        tmpcost = desired_outputs[i][j] - lay[num_layers-1].neu[j].actv;
        cost[j] = (tmpcost * tmpcost)/2;
        tcost = tcost + cost[j];
    }

    full_cost = (full_cost + tcost)/n;
    n++;
}

//反向传播
void back_prop(int p)
{
    int i,j,k;

    // Output Layer
    for(j=0;j<num_neurons[num_layers-1];j++)
    {
        lay[num_layers-1].neu[j].dz = (lay[num_layers-1].neu[j].actv - desired_outputs[p][j]) * (lay[num_layers-1].neu[j].actv) * (1- lay[num_layers-1].neu[j].actv);

        for(k=0;k<num_neurons[num_layers-2];k++)
        {
            lay[num_layers-2].neu[k].dw[j] = (lay[num_layers-1].neu[j].dz * lay[num_layers-2].neu[k].actv);
            lay[num_layers-2].neu[k].dactv = lay[num_layers-2].neu[k].out_weights[j] * lay[num_layers-1].neu[j].dz;
        }

        lay[num_layers-1].neu[j].dbias = lay[num_layers-1].neu[j].dz;
    }

    // Hidden Layers
    for(i=num_layers-2;i>0;i--)
    {
        for(j=0;j<num_neurons[i];j++)
        {
            if(lay[i].neu[j].z >= 0)
            {
                lay[i].neu[j].dz = lay[i].neu[j].dactv;
            }
            else
            {
                lay[i].neu[j].dz = 0;
            }

            for(k=0;k<num_neurons[i-1];k++)
            {
                lay[i-1].neu[k].dw[j] = lay[i].neu[j].dz * lay[i-1].neu[k].actv;

                if(i>1)
                {
                    lay[i-1].neu[k].dactv = lay[i-1].neu[k].out_weights[j] * lay[i].neu[j].dz;
                }
            }

            lay[i].neu[j].dbias = lay[i].neu[j].dz;
        }
    }
}

//测试网络
void test_nn(void)
{
    int i;
    while(1)
    {
        printf("Enter input to test:\n");

        for(i=0;i<num_neurons[0];i++)
        {
            scanf("%f",&lay[0].neu[i].actv);
        }
        forward_prop();
    }
}

int dinit(void)
{
    return SUCCESS_DINIT;
}