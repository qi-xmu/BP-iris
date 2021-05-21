#include <iostream>
#include "lib/Dateset/Dateset.h"
#include "lib/BPNet/BPNet.h"

using namespace std;

int main() {
    Dateset<double> d(R"(D:\2.code\Cpp\CLion\BP-iris\iris.data)");
    d.dataLoader();
    hiddenLayer l(3, 4);



}
