#include "Dataset.h"
#include "BPNet.h"
#include <ctime>

using namespace std;

int main() {
    Dataset<double> d("../seeds_dataset.csv", 7, '\t');
//    Dataset<double> d("../iris.data", 4, ',');

    d.dataLoader();          /* 加载数据 */
    d.normalize();           /* 数据归一化 */
    d.divide(3);        /* 数据划分 */
    d.confuse(1000);    /* 数据混淆：打乱顺序 */

    vector<v_double> data = d.Data();                  /* 全部数据集 */
    BPNet net(7, 3, 0.4);     /* 指定数据维度，分类数，学习率 */

    net.dataReader(d.train_data, d.eval_data);         /* 加载数据集 */

    /* 网络结构 */
//    net.addHiddenLayer(1);
//    net.addHiddenLayer(4);
//    net.summary();                /* 打印网络结构 */
//    clock_t start = clock();    /* 开始计时 */
//    net.train(600);             /* 开始训练 */
//    clock_t end = clock();      /* 结束计时 */
//    cout << "训练时间：" << (end - start) / (double) CLOCKS_PER_SEC << endl;
//    net.save("../model/best-seed-1.0.model");
//    net.evaluate();               /* 开始评估 */

    BPNet load_net(7,3,1);
    load_net.dataReader(d.train_data, data);
    load_net.load("../model/best-seed-1.0.model");
    load_net.evaluate();
    return 0;
}
