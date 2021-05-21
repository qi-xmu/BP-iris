#include <iostream>
#include "Dateset.h"
#include "BPNet.h"

using namespace std;

int main() {
    Dateset<double> d(R"(..\iris.data)");
    d.dataLoader();
    /* 数据混淆 */
    d.confuse(50);
    /* 指定数据维度，分类数，学习率 */
    BPNet net(4, 3, 0.1);
    /* 加载数据集 */
    net.dataReader(d.train_data, d.eval_data);
    /* 网络结构 */
    net.addHiddenLayer(4);
    net.addHiddenLayer(5);
    net.addHiddenLayer(6);
    net.addHiddenLayer(4);
    /* 打印网络结构 */
    net.summary();
    /* 开始训练 */
    net.train();
    /* 开始评估 */
    net.evaluate();

    return 0;
}
