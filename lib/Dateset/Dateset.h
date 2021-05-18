//
// Created by 36014 on 2021/5/18.
//

#ifndef BP_IRIS_DATESET_H
#define BP_IRIS_DATESET_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <ctime>

using namespace std;

template<class T>
class Dateset {
public:
    /* 数据集 */
    int dim;    // 数据维度
    vector<vector<T> > train_data;
    vector<vector<T> > eval_data;

    Dateset(const string &file_path, int dim = 4, int _mod = 10);

    /* 加载数据 */
    void dataLoader();

    /* 数据混淆 */
    void confuse(const int freq = 50);

private:
    string file_path;   // 文件路径
    int mod;            // 取测试集的mod
};


template<class T>
Dateset<T>::Dateset(const string &file_path, int dim, int mod) {
    this->file_path = file_path;
    this->mod = mod;
    this->dim = dim;
}

template<class T>
void Dateset<T>::dataLoader() {
    int cnt = 1;
    string line;
    ifstream fp(file_path);
    /* 逐行读取文件 */
    while (getline(fp, line)) {
        string item;
        vector<T> line_data;            /* 一行的数据 */
        istringstream num_str(line);    /* 字符串数据流化 */
        for (int i = 0; i < dim + 1; i++) {
            getline(num_str, item, ',');
            /* 标签识别 */
            if (item == "Iris-setosa")
                line_data.push_back(0);
            else if (item == "Iris-versicolor")
                line_data.push_back(1);
            else if (item == "Iris-virginica")
                line_data.push_back(2);
            else
                line_data.push_back(atof(item.c_str()));
        }
        /* 筛选train数据和eval数据 */
        if (cnt % mod == 0)
            eval_data.push_back(line_data);
        else
            train_data.push_back(line_data);
        cnt++;
    }
}

template<class T>
void Dateset<T>::confuse(const int freq) {
    time_t t;
    int len = train_data.size();
    srand((unsigned) time(&t));
    for (int i = 0; i < freq; i++) {
        int a = rand() % len;
        int b = rand() % len;
        train_data[a].swap(train_data[b]);
    }
}


#endif //BP_IRIS_DATESET_H
