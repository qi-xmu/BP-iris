//
// Created by 36014 on 2021/5/18.
//

#include "../include/BPNet.h"

#include <ctime>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>

using namespace std;

/*
 * Layer类 */

/* 设置初始值 */
void Layer::set(int dim) {
    this->node_num = dim;                           /* 数据维度即节点个数，偏置节点默认添加 */
    /* 初始化偏置节点 */
    this->node_value.resize(node_num + 1);  /* 包含偏置节点 */
    this->node_value[node_num] = 1;                 /* 偏置节点的输出始终为 1 */
}

/* 输入函数：节点数，节点值 */
void Layer::input(v_double &value) {
    this->node_value = value;
    this->node_value[node_num] = 1;                 /* 偏置量设置为 1 */
}

/* 输出函数 */
v_double Layer::output() {
    v_double out = node_value;
    out.pop_back();                                 /* 不输出偏置节点 */
    return out;
}

/*
 * hiddenLayer类 */

void hiddenLayer::set(int pre_node_num, int node_num) {
    /* 初始化节点数据 */
    this->pre_node_num = pre_node_num;              /* +1 偏置节点，应该对应value的dim */
    this->node_num = node_num;

    /* 初始化权重 */
    initWeight();
    /* 初始化节点 */
    node_value.resize(node_num + 1);        /* 权重节点+偏置节点 */
    node_value[node_num] = 1;
    /* 权重节点的输出始终为 1 */
}

/* 激活函数 */
inline double hiddenLayer::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
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
        w.resize(pre_node_num + 1);         /* +1 偏置 */
        /* 这里是每一个节点包含的向量 w ，数量为 前节点数量 + 偏置节点 */
        for (int j = 0; j <= pre_node_num; j++) {   /* = 偏置 */
            w[j] = (rand() % 100 + 50) / 150.0;
        }
        W[i] = w;
    }
}

/* 计算节点输出值 */
v_double hiddenLayer::calNodeValue(vector<double> value) {
    /* 更新节点的权重，偏置节点不更新 */
    for (int i = 0; i < node_num; i++) {
        for (int j = 0; j <= pre_node_num; j++) {   /* = 偏置 */
            node_value[i] += W[i][j] * value[j];
        }
        node_value[i] = sigmoid(node_value[i]);   /* 激活函数 */
    }
    return node_value;                              /* 节点输出值 */
}

/* 计算各个节点的残差（不包括输出层） */
void hiddenLayer::calNodeResidual(v_double value) {
    node_residual.resize(node_num + 1);     /* +1 偏置 */
    for (int i = 0; i <= node_num; i++) {           /* = 偏置 */
        /* 这里的 value = sum{下层网络残差 * 权值（由该节点指向下一层的权值）} */
        node_residual[i] = value[i] * (node_value[i] * (1 - node_value[i]));
    }
}

/* 反向传播值 残差 * 权值 */
v_double hiddenLayer::nodeBackValue() {
    /* 计算上一层残差需要的值 */
    v_double value(pre_node_num + 1);            /* +1 偏置节点 */
    for (int i = 0; i <= pre_node_num; i++)         /* = 偏置结点 */
        for (int j = 0; j < node_num; j++) {
            value[i] += node_residual[j] * W[j][i];
        }
    return value;                                   /* 返回值为反向传递值 */
}

/* 通过残差更新网络 */
void hiddenLayer::updateWeights(v_double pre_node_value, double learning_rate) {
    /* 更新输出层权重 */
    for (int i = 0; i < node_num; i++) {
        /* 更新一个节点内的权重，最后一个为偏置量 */
        for (int j = 0; j <= pre_node_num; j++) {   /* = 偏置 */
            double update_value = learning_rate * node_residual[i] * pre_node_value[j];
            W[i][j] -= update_value;
        }
    }
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
    return node_residual;
}

/* 总体误差 */
double hiddenLayer::totalError(int label) {
    double error = 0;
    /* 制作标签容器 */
    v_double lab(node_num, 0);
    lab[label] = 1;
    for (int i = 0; i < node_num; i++) {
        error += pow(node_value[i] - lab[i], 2);
    }
    return error;
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

/* 返回值：误差  反向传播 */
double BPNet::backward(int label) {
    /* 计算输出层残差 */
    output_layer.outputResidual(label);
    v_double value = output_layer.nodeBackValue();
    /* 更新输出层权重 */
    output_layer.updateWeights(
            hidden_layers[layers_num - 1].node_value,
            learning_rate);
    for (int i = (int) layers_num - 1; i >= 1; i--) {
        /* 计算第i层残差 */
        hidden_layers[i].calNodeResidual(value);
        /* 计算传递值 残差 * 权值 */
        value = hidden_layers[i].nodeBackValue();
        /* 更新权重 */
        hidden_layers[i].updateWeights(
                hidden_layers[i - 1].node_value,
                learning_rate);
    }
    /* 计算隐藏层第一层权重 */
    hidden_layers[0].calNodeResidual(value);
    /* 更新隐藏层第一层权重 */
    hidden_layers[0].updateWeights(
            input_layer.node_value,
            learning_rate);
    return output_layer.totalError(label);
}

/* 训练 */
double BPNet::train(int epoch) {
    cout << "Start to train >>>" << endl;
    int cnt = 1, right_cnt = 0;
    for (int i = 0; i < epoch; i++) {
        for (auto each : train_data) {
            forward(each);                               /* 前向传播 */
            v_double out = output_layer.output();
            printf("> %05d (", cnt);                    /* 训练编号 */
            for (auto it : out) {
                printf("%.6f, ", it);
            }
            printf(")\tError: %.6f", backward(each[dim]));    /* 误差反向传播 */
            int predict = findMax(out);
            cnt++;
            printf("\tLabel/Predict %d/%d\t", int(each[dim]), predict);
            if (predict == each[dim]) {
                right_cnt++;
                cout << "Right!";
            }
            cout << endl;
        }
    }
    double accuracy = right_cnt / (cnt - 1.0);
    cout << "End trained. Accuracy: " << accuracy << "<<<" << endl;
    return accuracy;
}

/* 预测 */
double BPNet::evaluate() {
    cout << "Start to evaluate >>>" << endl;
    int cnt = 1, right_cnt = 0;
    for (auto each : test_data) {
        forward(each);
        v_double out = output_layer.output();
        printf("> %03d (", cnt);
        for (auto it : out) {
            printf("%.6f, ", it);
        }
        int predict = findMax(out);
        printf(")\tLabel/Predict %d/%d\t", int(each[dim]), predict);
        if (predict == each[dim]) {
            right_cnt++;
            cout << "Right!";
        }
        cnt++;
        cout << endl;
//        backward(each[dim]);
    }
    double accuracy = right_cnt / (cnt - 1.0);
    cout << "End evaluated. Accuracy: " << accuracy << "<<<" << endl;
    return accuracy;
}

/* 输出网络结构 */
void BPNet::summary() {
    printf("输入层：\t%d个节点\n", input_layer.nodeSize());
    cout << "隐藏层：" << endl;
    for (int i = 0; i < layers_num; i++) {
        printf("\t第%d层：\t%d个节点\n", i + 1,
               hidden_layers[i].nodeSize());
    }
    printf("输出层：\t%d个节点\n", output_layer.nodeSize());
};

/* 模型保存 */
void BPNet::save(const string &path) {
    fstream outfile;
    outfile.open(path, ios::out);
    outfile << layers_num << endl;
    int pre_node_size = input_layer.nodeSize();
    for (int i = 0; i < layers_num; i++) {
        int node_size = hidden_layers[i].nodeSize();        /* 节点个数 */
        outfile << node_size << " ";
        outfile << pre_node_size + 1 << endl;               /* 前一层节点数 +1 偏置节点 */
        vector<v_double > Weights = hidden_layers[i].nodeWeights();
        for (int j = 0; j < node_size; j++) {
            for (int k = 0; k <= pre_node_size; k++) {     /* = 偏置 */
                outfile << setprecision(10)<< setw(12) << Weights[j][k] << " ";
            }
            outfile << endl;
        }
        pre_node_size = hidden_layers[i].nodeSize();
    }
    /* 输出层权重 */
    int node_size = output_layer.nodeSize();
    outfile << node_size << " ";
    outfile << pre_node_size + 1 << endl;
    vector<v_double > Weights = output_layer.nodeWeights();
    for (int j = 0; j < num_classes; j++) {
        for (int k = 0; k <= pre_node_size; k++) {     /* = 偏置 */
            outfile << setprecision(10) << setw(12) << Weights[j][k] << " ";
        }
        outfile << endl;
    }
    cout << "Save Finished to " << path << endl;
}

/* 模型加载 */
void BPNet::load(const string &path) {
    ifstream infile(path, ios::in);
    /* 打开文件失败 */
    if (!infile.is_open()) {
        cout << "Error opening file";
        return;
    }
    /* 打开文件成功 */
    int layer_size;                     /* 隐藏层层数 */
    infile >> layer_size;
    for (int i = 0; i < layer_size; i++) {
        int node_num, weight_num;       /* 节点个数 权重个数 */
        infile >> node_num >> weight_num;
        addHiddenLayer(node_num);       /* 新建一层，已经更新权重 */
        for (int j = 0; j < node_num; j++) {
            for (int k = 0; k < weight_num; k++) {
                infile >> hidden_layers[i].W[j][k];
            }
        }
    }
    /* 输出层单独加载 */
    int node_num, weight_num;
    infile >> node_num >> weight_num;
    for (int j = 0; j < node_num; j++) {
        for (int k = 0; k < weight_num; k++) {
            infile >> output_layer.W[j][k];
        }
    }
}

/*  寻找最大值  */
int BPNet::findMax(v_double x) {
    int key = 0;
    double max = x[0];
    for (int i = 1; i < x.size(); i++) {
        if (x[i] > max) {
            max = x[i], key = i;
        }
    }
    return key;
}




