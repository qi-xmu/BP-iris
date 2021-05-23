#include "Dateset.h"
#include "BPNet.h"
#include <ctime>

using namespace std;

int main() {
    Dateset<double> d(R"(..\iris.data)");
    d.dataLoader();
    /* 数据归一化 */
    d.normalize();
    d.divide(3);
    /* 数据混淆：打乱顺序 */
    d.confuse(1000);
    /* 指定数据维度，分类数，学习率 */
    BPNet net(4, 3, 1);
    /* 加载数据集 */
    net.dataReader(d.train_data, d.eval_data);
    /* 网络结构 */
    net.addHiddenLayer(4);
    net.addHiddenLayer(4);
    /* 打印网络结构 */
    net.summary();
    /* 开始训练 */
    clock_t start = clock();    /* 开始计时 */
    net.train(  400);
    clock_t end = clock();      /* 结束计时 */
    /* 开始评估 */
    net.evaluate();
    cout << "训练时间：" << (end - start) /(double)CLOCKS_PER_SEC << endl;

    return 0;
}
