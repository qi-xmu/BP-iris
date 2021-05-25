//
// Created by 36014 on 2021/5/18.
//

/* 神经元层类 神经网络层设计
 * * 基类 Layer 属性
 * 值：节点数量，各个节点权重
 * 行为：构造，输入，输出
 *
 * * 派生类 inputLayer 属性
 * 行为：输入
 *
 * * 派生类 hiddenLayer 属性
 * 值：前网络层节点个数
 * 行为：计算，更新，激活函数
 *
 * * 派生类 outputLayer 属性
 * 行为：计算残差
 *
 * 将 inputLayer 合并到 Layer 中
 * 将 outputLayer 合并到 hiddenLayer 中
 * */

#ifndef BP_IRIS_BPNET_H
#define BP_IRIS_BPNET_H

#include <vector>
#include <string>
#include <fstream>

#define v_double vector<double>

using namespace std;

class Layer {
public:
    v_double node_value;                /* 节点输出值 */
    vector<v_double > W;                /* 权重，public这里提供外部修改接口 */

    /* 设置初始值 */
    void set(int dim);

    /* 输入函数 */
    void input(v_double &value);
    /* 输出函数 */
    v_double output();

    /* 返回权重节点个数 不包括偏置节点 */
    int nodeSize() const { return node_num; };

    /* 保存使用:到处网络权重值 */
    vector<v_double > nodeWeights() const { return W; };

protected:
    int node_num;                       /* 网络层权重节点个数，不包含偏置节点 */
};

/* 隐藏神经元层 */
class hiddenLayer : public Layer {
public:
    v_double node_residual;             /* 节点残差 */

    /* 设置网络层参数 */
    void set(int pre_node_num, int node_num);

    /* 初始化权值 */
    void initWeight();
    /* 计算节点输出值并返回 */
    v_double calNodeValue(vector<double> value);

    /* 计算各个节点的残差 */
    void calNodeResidual(v_double value);
    /* 返回节点传递计算值 */
    v_double nodeBackValue();

    /* 更新权值和偏置 */
    void updateWeights(v_double pre_node_value,
                       double learning_rate);
    /* 输出层：计算输出层残差 */
    v_double outputResidual(int label);

    /* 输出层：计算误差值 */
    double totalError(int label);

private:
    int pre_node_num;                   /* 上一层节点数 */
    inline double sigmod(double x);     /* 激活函数 */
};

class BPNet {
public:
    /* 初始化输入层，输出层 */
    BPNet(int dim, int num_classes, double learning_rate = 0.5);

    /* 增加隐藏层 */
    void addHiddenLayer(int node_num);

    /* 读取数据 数据格式： 二维数组 vector<v_double > 数据 标签 */
    void dataReader(const vector<v_double > &train_data,
                    const vector<v_double > &test_data);

    /* 训练 */
    double train(int epoch = 1);

    /* 预测 */
    double evaluate();

    /* 打印当前网络结构 */
    void summary();

    /* 模型保存 */
    void save(const string &path);

    /* 模型加载 */
    void load(const string &path);

private:
    int dim;                            /* 输入数据维度 */
    int num_classes;                    /* 分类数量 */
    double learning_rate;               /* 学习率 */
    double total_error;                 /* 总误差 */

    unsigned long long layers_num;      /* 隐藏层个数 */

    /* 没有采用数据流的方式，这里选择直接读取所有数据 */
    vector<v_double > train_data;       /* 训练集数据 */
    vector<v_double > test_data;        /* 测试集数据 */

    Layer input_layer;                  /* 输入层 */
    vector<hiddenLayer> hidden_layers;  /* 隐藏层 */
    hiddenLayer output_layer;           /* 输出层 */

    /* 前向传播 */
    void forward(v_double value);

    /* 反向传播 */
    double backward(int label);

    /* 寻找最大值下标 */
    int findMax(v_double x);
};


#endif //BP_IRIS_BPNET_H
