# BP神经网路-鸢尾花分类

 **XMU @2021 QI 面向对象C++课程作业**

### 项目完成情况

- [x] 自定义数据。包含数据加载，数据归一化，数据混淆，数据划分。
- [x] 自定义模型结构。建议隐藏层不超过两层，每一层节点不建议过多，否则可能造成梯度爆炸或梯度消失。
- [x] 自定义训练轮回。建议值200~1000。训练次数不是越多越好，适当的训练次数可以节省时间。
- [x] 完成模型保存和加载。提高模型保存精度，保证导出模型再加载产生的舍入误差较小。

## 文件结构

```
|--build/release 		 # 编译缓冲文件
|	 --libBPNet.a      # 静态库，可以再其他项目中引用
|--include
|	 --BPNet.h		
|	 --Dataset.h
|--model
|	 --best-5.0.model  # 最佳模型 准确率100%
|	 --...
|--src
|	 --BPNet.cpp   		 # BPNet源码
|	 --Dataset.cpp     # Dataset较小，直接再头文件中定义
|--main.cpp					 # 主程序文件
|--CmakeList.txt     # 项目编译，可以用cmake编译
|--iris.data         # 数据集
```

### 团队成员

[Qi](https://github.com/qi-xmu) | [asd523303](https://github.com/asd52303) 

