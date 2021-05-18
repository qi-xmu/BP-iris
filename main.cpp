#include <iostream>
#include "lib/Dateset/Dateset.h"

using namespace std;

int main() {
    Dateset<double> d(R"(D:\2.code\Cpp\CLion\BP-iris\iris.data)");
    d.dataLoader();
    cout <<"ÑµÁ·¼¯" << endl;
    for(auto line :d.train_data){
        for (auto num : line){
            cout << num << "\t";
        }
        cout << endl;
    }
    cout <<"²âÊÔ¼¯" << endl;
    for(auto line :d.eval_data){
        for (auto num : line){
            cout << num << "\t";
        }
        cout << endl;
    }
    d.confuse(50);

    cout <<"ÑµÁ·¼¯" << endl;
    for(int i=0;i<135;i++){
        for (int j=0;j<5;j++){
            cout << d.train_data[i][j] << "\t";
        }
        cout << endl;
    }
}
