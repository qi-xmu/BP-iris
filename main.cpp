#include <iostream>
#include "lib/Dateset/Dateset.h"
#include "lib/BPNet/BPNet.h"

using namespace std;

int main() {
    Dateset<double> d(R"(D:\2.code\Cpp\CLion\BP-iris\iris.data)");
    d.dataLoader();
    /* ָ������ά�ȣ���������ѧϰ�� */
    BPNet net(4, 3, 0.01);
    /* �������ݼ� */
    net.dataReader(d.train_data, d.eval_data);







}
