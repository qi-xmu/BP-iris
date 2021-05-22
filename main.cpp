#include <iostream>
#include "Dateset.h"
#include "BPNet.h"

using namespace std;

int main() {
    Dateset<double> d(R"(..\iris.data)");
    d.dataLoader();

    d.normalize();
    d.divide(10);
    /* 数据混淆 */
//    d.confuse(10000);

    /* 指定数据维度，分类数，学习率 */
    BPNet net(4, 3, 1);
    /* 加载数据集 */
    net.dataReader(d.train_data, d.eval_data);
    /* 网络结构 */
    net.addHiddenLayer(4);
    net.addHiddenLayer(6);
    net.addHiddenLayer(8);
    net.addHiddenLayer(10);
    net.addHiddenLayer(10);
    net.addHiddenLayer(8);
    net.addHiddenLayer(6);
    net.addHiddenLayer(4);
    net.addHiddenLayer(3);
    /* 打印网络结构 */
    net.summary();
    /* 开始训练 */
    net.train(100);
    /* 开始评估 */
    net.evaluate();

    return 0;
}
