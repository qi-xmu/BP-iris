//
// Created by 36014 on 2021/5/18.
//

#ifndef BP_IRIS_DATESET_H
#define BP_IRIS_DATESET_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

template<class T>
class Dateset {
public:
    /* 数据集 */
    int len;    // 数据集长度
    int dim;    // 数据维度
    vector<vector<T> > train_data;
    vector<vector<T> > eval_data;

    Dateset(string file_path, int dim = 4, int _mod = 10);

    /* 加载数据 */
    void dataLoader();

private:
    string file_path;   // 文件路径
    int mod;            // 取测试集的mod
};


template<class T>
Dateset<T>::Dateset(string _file_path, int _dim, int _mod) {
    this->file_path = _file_path;
    this->mod = _mod;
    this->dim = _dim;
}

template<class T>
void Dateset<T>::dataLoader() {
    int cnt = 1;
    string line;
    ifstream fp(file_path);
    /* 逐行读取文件 */
    while (getline(fp, line)) {
        string num;
        vector<T> line_data;            /* 一行的数据 */
        istringstream num_str(line);    /* 字符串数据流化 */
        for (int i = 0; i < 5; i++) {
            getline(num_str, num, ',');
            line_data.push_back(atof(num.c_str()));
        }
        /* 筛选train数据和eval数据 */
        if (cnt % mod == 0)
            eval_data.push_back(line_data);
        else
            train_data.push_back(line_data);
        cnt++;
    }
}


#endif //BP_IRIS_DATESET_H
