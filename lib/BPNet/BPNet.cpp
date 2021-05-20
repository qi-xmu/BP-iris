//
// Created by 36014 on 2021/5/18.
//

#include "BPNet.h"

#include <ctime>
#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace std;

/* 构造函数 */
Layer::Layer() {};

Layer::Layer(int node_num, vector<double> value) {
    this->node_num = node_num;
    this->node_value = value;
}

/* 输入函数：节点数，节点值 */
void Layer::input(int dim, v_double &value) {
    this->node_num = dim;
    this->node_value = value;
}

/* 输出函数 */
v_double Layer::output() {
    return node_value;
}

/* 隐藏层构造函数 */
hiddenLayer::hiddenLayer(int pre_node_num, int node_num) {
    /* 要多一个，保证有偏置值。 */
    this->pre_node_num = pre_node_num;
    this->node_num = node_num;
    initWeight();
}

/* 初始化神经网络层权重 */
void hiddenLayer::initWeight() {
    /* i个节点，每个节点有j个权重 */
    for (int i = 0; i < node_num; i++) {
        v_double w;
        w.reserve(pre_node_num);
        for (int j = 0; j < pre_node_num; j++) {
            /*  w.i.j  -100.00 ~ 100.00 */
            w.push_back((rand() % 10000) / 100.0);
        }
        W.push_back(w);
    }
}

/* 计算节点输出值 */
v_double hiddenLayer::getNodeValue(v_double &value) {
    /* 计算一个节点的权重 */
    for (int i = 0; i < node_num; i++) {
        for (int j = 0; j < node_num; j++) {
            node_value[i] += W[i][j] * value[j];
        }
        sigmod(node_value[i]);
    }
    return node_value;
}

/* 激活函数 */
void hiddenLayer::sigmod(double &x) {
    x = 1.0 / (1.0 + exp(-x));
}




