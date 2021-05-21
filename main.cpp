#include <iostream>
#include "lib/Dateset/Dateset.h"
#include "lib/BPNet/BPNet.h"

using namespace std;

int main() {
    Dateset<double> d(R"(D:\2.code\Cpp\CLion\BP-iris\iris.data)");
    d.dataLoader();
    /* 指定数据维度，分类数，学习率 */
    BPNet net(4, 3, 0.01);
    /* 加载数据集 */
    net.dataReader(d.train_data, d.eval_data);







}
