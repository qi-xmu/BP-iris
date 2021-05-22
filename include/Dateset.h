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
    int dim{};    // 数据维度
    vector<vector<T> > train_data;  /* 训练集数据 */
    vector<vector<T> > eval_data;   /* 测试集数据 */

    explicit Dateset(const string &file_path, int dim = 4);
    /* 加载数据 */
    void dataLoader();
    /* 数据归一化 */
    void normalize();
    /* 划分数据集 */
    void divide(int mod = 10);
    /* 数据混淆 */
    void confuse(int freq = 50);

    void print();

private:
    string file_path;                   // 文件路径
    vector<vector<T> > data;            // 所有文件数据
    vector<T> max_value, min_value;     /* 归一化：最大最小值 */
};

template<class T>
Dateset<T>::Dateset(const string &file_path, int dim) {
    this->file_path = file_path;
    this->dim = dim;
}

/* 加载数据集 */
template<class T>
void Dateset<T>::dataLoader() {
    /* 归一化准备
     * 寻找最大值 最小值 */
    max_value.resize(dim, 0);
    min_value.resize(dim, 100000000);
    string line, item;
    ifstream fp(file_path);                         /* 加载文件流 */
    while (getline(fp, line)) {
        vector<T> line_data;                        /* 一行的数据 */
        istringstream num_str(line);                /* 字符串数据流化 */
        for (int i = 0; i < dim + 1; i++) {
            getline(num_str, item, ',');

            if (i == dim) {
                if (item == "Iris-setosa")              /* 种类一 标签 0 */
                    line_data.push_back(0);
                else if (item == "Iris-versicolor")     /* 种类二 标签 1 */
                    line_data.push_back(1);
                else if (item == "Iris-virginica")      /* 种类三 标签 2 */
                    line_data.push_back(2);
            }
            else {
                /* 储存最大值和最小值 */
                T value = atof(item.c_str());
                max_value[i]  = (max_value[i] < value)? value : max_value[i];
                min_value[i]  = (min_value[i] > value)? value : min_value[i];

                line_data.push_back(value);
            }
        }
        data.push_back(line_data);
    }
}

/* 归一化 */
template<class T>
void Dateset<T>::normalize() {
    for(int i=0;i<data.size();i++){
        for(int j =0;j<dim;j++){
            data[i][j] = (max_value[j] - data[i][j]) / (max_value[j] - min_value[j]);
        }
    }
}
/* 划分数据集 */
template<class T>
void Dateset<T>::divide(int mod) {
    for(int i =0; i < data.size();i++){
        if(i % mod == 0)
            eval_data.push_back(data[i]);
        else
            train_data.push_back(data[i]);
    }
}

template<class T>
void Dateset<T>::confuse(int freq) {
    time_t t;
    int len = train_data.size();
    if(len == 0){
        cout << "ERR Length = 0!";
        return;
    }
    srand((unsigned) time(&t));
    for (int i = 0; i < freq; i++) {
        int a = rand() % len;
        int b = rand() % len;
        train_data[a].swap(train_data[b]);
    }
}

template<class T>
void Dateset<T>::print() {
    for(auto each : data){
        cout << "> ";
        for(auto it : each){
            cout << it << "\t";
         }
        cout << endl;
    }
}



#endif //BP_IRIS_DATESET_H