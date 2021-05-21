//
// Created by 36014 on 2021/5/18.
//

#include "BPNet.h"

#include <ctime>
#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace std;

/*
 * Layer类 */

/* 设置初始值 */
void Layer::set(int dim) {
    this->node_num = dim;   /* 数据维度即节点个数，偏置节点自行添加 */
    /* 初始化偏置节点 */
    this->node_value.resize(node_num + 1);
    this->node_value[node_num] = 1;  /* 偏置节点的输出始终为 1 */
}

/* 输入函数：节点数，节点值 */
void Layer::input(v_double &value) {
    this->node_value = value;
    this->node_value[node_num] = 1;  /* 偏置量设置为 1 */
}

/* 输出函数 */
v_double Layer::output() {
    v_double out = node_value;
    out.pop_back();
    return out;
}

/* 反向传播值 残差 * 权值 */
v_double hiddenLayer::nodeBackValue() {
    /* 计算上一层残差需要的值 */
    v_double value(pre_node_num);
    for (int i = 0; i < pre_node_num; i++)
        for (int j = 0; j < node_num; j++) {
            value[i] += node_residual[j] * W[j][i];
        }
    return value;
}


/*
 * hiddenLayer类 */

void hiddenLayer::set(int pre_node_num, int node_num) {
    /* 初始化节点数据 */
    this->pre_node_num = pre_node_num;  /* +1 偏置节点，应该对应value的dim */
    this->node_num = node_num;

    /* 初始化权重 */
    initWeight();
    /* 初始化节点 */
    node_value.resize(node_num + 1);  /* 权重节点+偏置节点 */
    node_value[node_num] = 1;
    /* 权重节点的输出始终为 1 */
}


/* 初始化神经网络层权重 */
void hiddenLayer::initWeight() {
    /* i个节点，每个节点有j个权重 */
    time_t t;
    srand((unsigned int) time(&t));
    /* 对每一个节点权重取随机值 */
    W.resize(node_num);
    for (int i = 0; i < node_num; i++) {
        v_double w;
        /* 预分配内存，不改变size resize会改变size */
        w.resize(pre_node_num + 1);
        for (int j = 0; j < pre_node_num + 1; j++) {
            /*  w.i.j  -0.2000 ~ 0.2000 */
            w[j] = (rand() % 10000) / 50000.0;
        }
        W[i] = w;
    }
}

/* 计算节点输出值 */
v_double hiddenLayer::calNodeValue(vector<double> value) {
    /* 更新节点的权重，偏置节点不更新 */
    for (int i = 0; i < node_num; i++) {
        for (int j = 0; j < pre_node_num + 1; j++) {
            node_value[i] += W[i][j] * value[j];
        }
        sigmod(node_value[i]);  /* 激活函数 */
    }
    return node_value;
}

/* 激活函数 */
void hiddenLayer::sigmod(double &x) {
    x = 1.0 / (1.0 + exp(-x));
}

/* 计算各个节点的残差 */
void hiddenLayer::calNodeResidual(v_double value) {
    node_residual.resize(node_num);
    for (int i = 0; i < node_num; i++) {
        /* 这里的 value = sum{下层网络残差 * 权值（由该节点指向下一层的权值）} */
        node_residual[i] = value[i] * (node_value[i] * (1 - node_value[i]));
    }
    for (int i = 0; i < node_num; i++)
        cout << node_residual[i] << " ";
    cout << endl;
}

void hiddenLayer::updateWeights(v_double pre_node_value, double learning_rate) {
    /* 更新输出层权重 */
    v_double residual = node_residual;
    for (int i = 0; i<node_num;i++){
        for(int j=0; j<pre_node_num+1;j++){
            double update_value =  learning_rate * node_residual[i] * pre_node_value[j];
            W[i][j] -= update_value;
        }
    }
}

/*
 * BPNet类 */

/* 网络构造函数 */
BPNet::BPNet(int dim, int num_classes, double learning_rate) {
    this->dim = dim;                        /* 数据维度 */
    this->num_classes = num_classes;        /* 分类数量 */
    this->learning_rate = learning_rate;    /* 学习率 */
    this->layers_num = 0;                   /* 隐藏层层数 */

    /* 创建输入层 */
    input_layer.set(dim);   /* 设置数据的维度，网络层会自动添加偏置节点 */
    /* 创建输入层  这里的前节点个数不包含偏置节点 */
    output_layer.set(input_layer.nodeSize(), num_classes);
}

void BPNet::addHiddenLayer(int node_num) {
    hiddenLayer new_hidden_layer;   /* 需要添加的新隐藏层 */
    /* 连接前后层 */
    if (layers_num == 0) {   /* 当还没有隐藏层时，获取输入层参数 */
        new_hidden_layer.set(input_layer.nodeSize(),
                             node_num);
    } else {                 /* 当已经存在隐藏层时，获取最后一层参数 */
        new_hidden_layer.set(hidden_layers[layers_num - 1].nodeSize(),
                             node_num);
    }
    /* 添加 */
    hidden_layers.push_back(new_hidden_layer);
    /* 更新隐藏层层数 */
    layers_num = hidden_layers.size();
    /* 更新输出层参数 */
    output_layer.set(node_num, num_classes);
}

void BPNet::dataReader(const vector<v_double > &train, const vector<v_double > &test) {
    this->train_data = train;
    this->test_data = test;
}

/* 前向传播 */
void BPNet::forward(v_double value) {
    input_layer.input(value);
    /* 向前传递 */
    v_double layer_value = input_layer.output();
    for (int i = 0; i < layers_num; i++) {
        layer_value = hidden_layers[i].calNodeValue(layer_value);
    }
    /* 最后的结果存储在output_layer的node_value中 */
    output_layer.calNodeValue(layer_value);
}

/* 反向传播 */
void BPNet::backward(int label) {
    /* 计算输出层残差 */
    output_layer.outputResidual(label);
    v_double value = output_layer.nodeBackValue();
    /* 更新输出层权重 */
    output_layer.updateWeights(
            hidden_layers[layers_num-1].node_value,
            learning_rate);
    for (int i = (int) layers_num - 1; i >= 1; i--) {
        /* 计算第i层残差 */
        hidden_layers[i].calNodeResidual(value);
        /* 计算传递值 残差 * 权值 */
        value = hidden_layers[i].nodeBackValue();

        /* 更新权重 */
        hidden_layers[i].updateWeights(
                hidden_layers[i-1].node_value,
                learning_rate);
    }
    /* 计算隐藏层第一层权重 */
    hidden_layers[0].calNodeResidual(value);
    /* 更新隐藏层第一层权重 */
    hidden_layers[0].updateWeights(
            input_layer.node_value,
            learning_rate);
}

/* 计算输出节点残差
 * 每一个节点都有一个残差值 */
v_double hiddenLayer::outputResidual(int label) {
    /* 制作标签容器 */
    v_double lab(node_num, 0);
    lab[label] = 1;

    /* 求输出残差 */
    node_residual.resize(node_num);
    for (int i = 0; i < node_num; i++) {
        /* 公式：（预测 - 实际）* 预测函数的导数 f`(z_i)
         * f`(z_i) = f(z_i) * (1 - f(z_i)) */
        node_residual[i] = (node_value[i] - lab[i]) * (node_value[i] * (1 - node_value[i]));
    }
    cout << "\nResidual: ";
    for (auto each : node_residual)
        cout << each << " ";
    cout << endl;
    return node_residual;
}

/* 训练 */
void BPNet::train() {

}

/* 预测 */
void BPNet::evaluate() {

}

void BPNet::summary() {
    printf("输入层：\t%d个节点\n", input_layer.nodeSize());
    cout << "隐藏层：" << endl;
    for (int i = 0; i < layers_num; i++) {
        printf("\t第%d层：\t%d个节点\n", i + 1,
               hidden_layers[i].nodeSize());
    }
    printf("输出层：\t%d个节点\n", output_layer.nodeSize());
};

v_double BPNet::test() {
    /* 输入 -> 前向传播 -> 输出 */
    forward(train_data[0]);
    backward(0);
    return output_layer.output();
};