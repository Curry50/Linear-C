# Linear-C
## 基于C语言的全连接神经网络
### 一.运行
1.运行环境为Windows Clion 2022.1.3,直接运行即可  
2.运行环境为Windows Visual Studio Code + GCC 11.2.0,点击左下角蓝色工具栏Build后即可运行  
3.训练完成后需输入测试集进行测试，测试集各组数据均需*0.1后方可作为输入，具体原因见注意事项
### 二.分析
#### 1.参数修改
在main函数中找到对应变量，修改变量值
```
int num_layers;         //网络层数
int *num_neurons;       //各层神经元个数
float alpha;            //学习率
float *cost;            //损失
float full_cost;        //总损失
float **input;          //网络输入
float **desired_outputs;//数据标签
int num_training_ex;    //样本数
```
#### 2.网络输入
1.本代码使用banknote authentication数据集作为网络输入  
2.该数据集用于判断钞票真假，前四组数据为输入，最后一组数据为对应标签0/1  
#### 3.网络结构
1.由于输入数据有四组，因此输入层（第一层）需要有4个神经元  
2.初步设置网络有4层，各层神经元数为4、8、4、1  
3.非输出层（前三层）的激活函数为Relu，输出层的激活函数为sigmoid
#### 4.文件解析
1.layer.h创建层结构体，layer.c创建层变量，并对其参数进行初始化
```
layer create_layer(int number_of_neurons) //创建层
{
	layer lay;
	lay.num_neu = -1;
	lay.neu = (struct neuron_t *) malloc(number_of_neurons * sizeof(struct neuron_t));
	return lay;
}
```
2.neuron.h创建神经元结构体，neuron.c创建神经元变量，并对其参数进行初始化
```
neuron create_neuron(int num_out_weights) //创建神经元
{
	neuron neu;

	neu.actv = 0.0;
	neu.out_weights = (float*) malloc(num_out_weights * sizeof(float));
	neu.bias=0.0;
	neu.z = 0.0;

	neu.dactv = 0.0;
	neu.dw = (float*) malloc(num_out_weights * sizeof(float));
	neu.dbias = 0.0;
	neu.dz = 0.0;

	return neu;
}
```
3.backprop.h及backprop.c对以下函数进行了声明和定义
```
int create_architecture(void);    //构建网络结构
int initialize_weights(void);     //随机初始化参数
void feed_input(int i);           //导入数据
void train_neural_net(void);      //训练网络
void forward_prop(void);          //前向传播
void compute_cost(int i);         //计算损失
void back_prop(int p);            //反向传播
void update_weights(void);        //更新参数
void get_inputs(void);            //获得数据
void get_desired_outputs(void);   //获得标签
void test_nn(void);               //测试网络
```
#### 5.注意事项
由于输出层的激活函数为sigmoid  
sigmoid函数在变量取绝对值非常大的正值或负值时会出现饱和现象，意味着函数会变得很平，并且对输入的微小改变会变得不敏感   
在反向传播时，当梯度接近于0，权重基本不会更新，很容易就会出现梯度消失的情况，从而无法完成深层网络的训练  
因此在导入数据时，将各组数据均*0.1  
### 三.疑惑
### 四.参考
<https://github.com/mayurbhole/Neural-Network-framework-using-Backpropogation-in-C>
