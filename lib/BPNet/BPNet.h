//
// Created by 36014 on 2021/5/18.
//

#ifndef BP_IRIS_BPNET_H
#define BP_IRIS_BPNET_H

#include <vector>

#define v_double vector<double>

using namespace std;

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

class Layer {
public:
    Layer();

    Layer(int node_num, v_double value);

    /* 输入函数 */
    void input(int dim, v_double &value);
    /* 输出函数 */
    v_double output();

protected:
    vector<v_double > W;    /* 权重 */
    vector<v_double > R;    /* 残差 */
    v_double node_value;    /* 节点输出值 */
    int node_num;           /* 网络层节点个数 */
};

/* 隐藏神经元层 */
class hiddenLayer : public Layer {
public:
    hiddenLayer(int pre_node_num, int node_num);

    /* 初始化权值 */
    void initWeight();
    /* 计算节点值并返回 */
    v_double getNodeValue(v_double &value);

private:
    int pre_node_num;           /* 前节点个数，用于标记权重的维度 */
    void sigmod(double &x);     /* 激活函数 */
};


class BPNet {
public:

private:
    Layer input_layer;                  /* 输入层 */
    vector<hiddenLayer> hidden_layers;  /* 隐藏层 */
    hiddenLayer output_layer;           /* 输出层 */
};


#endif //BP_IRIS_BPNET_H
