#include "../include/layer.h"


layer create_layer(int number_of_neurons) //创建层
{
	layer lay;
	lay.num_neu = -1;
	lay.neu = (struct neuron_t *) malloc(number_of_neurons * sizeof(struct neuron_t));
	return lay;
}

// TODO:
// Add destructor
