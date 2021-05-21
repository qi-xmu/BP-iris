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

    net.addHiddenLayer(4);
    net.addHiddenLayer(5);
    net.addHiddenLayer(4);

    net.summary();

    cout << "Test: ";
    v_double out = net.test();
    for(int i=0;i<3;i++)
        cout << out[i] << " ";
    cout << "Label: " << d.train_data[0][4] << endl;

    cout << net.totalError() << endl;


    return 0;
}
