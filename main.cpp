#include "Dateset.h"
#include "BPNet.h"
#include <ctime>

using namespace std;

int main() {
    Dateset<double> d(R"(..\iris.data)");
    d.dataLoader();

    d.normalize();
    d.divide(5);
    /* 数据混淆 */
    d.confuse(1000);

    /* 指定数据维度，分类数，学习率 */
    BPNet net(4, 3, 1);
    /* 加载数据集 */
    net.dataReader(d.train_data, d.eval_data);
    /* 网络结构 */
    net.addHiddenLayer(3);
    /* 打印网络结构 */
    net.summary();
    /* 开始训练 */
    clock_t start = clock();
    net.train(100);
    clock_t end = clock();
    cout << "训练时间：" << (end - start) /(double)CLOCKS_PER_SEC << endl;
    /* 开始评估 */
    net.evaluate();

    return 0;
}
