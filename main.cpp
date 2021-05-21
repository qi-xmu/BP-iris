#include <iostream>
#include "lib/Dateset/Dateset.h"
#include "lib/BPNet/BPNet.h"

using namespace std;

int main() {
    Dateset<double> d(R"(D:\2.code\Cpp\CLion\BP-iris\iris.data)");
    d.dataLoader();
    /* 数据混淆 */
    d.confuse(50);
    /* 指定数据维度，分类数，学习率 */
    BPNet net(4, 3, 0.1);
    /* 加载数据集 */
    net.dataReader(d.train_data, d.eval_data);

    net.addHiddenLayer(4);
    net.addHiddenLayer(5);
    net.addHiddenLayer(6);
    net.addHiddenLayer(4);

    net.summary();

    net.train();

    net.evaluate();

    return 0;
}
