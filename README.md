### 开始之前

* 实验环境：Ubuntu 16.04/Caffe/1080 Ti
* 实验数据
数据来源: [Stanford Cars Dataset](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)
>
```
    -cars_meta.mat:
      Contains a cell array of class names, one for each class.

    -cars_train_annos.mat:
      Contains the variable 'annotations', which is a struct array of length
      num_images and where each element has the fields:
        bbox_x1: Min x-value of the bounding box, in pixels
        bbox_x2: Max x-value of the bounding box, in pixels
        bbox_y1: Min y-value of the bounding box, in pixels
        bbox_y2: Max y-value of the bounding box, in pixels
        class: Integral id of the class the image belongs to.
        fname: Filename of the image within the folder of images.

    -cars_test_annos.mat:
      Same format as 'cars_train_annos.mat', except the class is not provided.
```


[Stanford Cars Dataset](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)这个数据集一共196类，训练集8144，测试集8041。`cars_meta.mat`包含了所有类的编号和类名，`cars_train_annos.mat`包含了训练集车的坐标，文件名，`cars_test_annos.mat`同理。`cars_test_annos_withlabels.mat`另外包含了测试集的编号，用于测试。文件可在data下找到。
    


<div align="center">
  <img src="http://omoitwcai.bkt.clouddn.com/2017-11-07-%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202017-11-07%20%E4%B8%8B%E5%8D%884.24.11-1.png">
</div>
<div align="center">
  <img src="http://omoitwcai.bkt.clouddn.com/2017-11-07-%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202017-11-07%20%E4%B8%8B%E5%8D%884.23.57-1.png">
</div>

* GitHub现存别人做好的有：[ChenJoya/Vehicle_Detection_Recognition](https://github.com/ChenJoya/Vehicle_Detection_Recognition), [isapient/caffenet-stanford-cars](https://github.com/isapient/caffenet-stanford-cars)。 第一个没有准确度，第二个效果不好，没有裁剪出车辆，另外前两次没有分清训练集和测试集。

* 论文原始链接: [3D Object Representations for Fine-Grained Categorization ](https://www.cv-foundation.org/openaccess/content_iccv_workshops_2013/W19/html/Krause_3D_Object_Representations_2013_ICCV_paper.html), 顶会顶会！！！

### 思路
数据预处理 ——> 训练（Fine-tuning on AlexNet, CaffeNet,训练代价小，不容易过拟合）——> 预测
#### data preprocessing
* 裁剪
根据bouding box裁剪出车辆
* 重采样
裁剪出车辆之后，图片大小不一致，需要将裁减后的图片重新采样到同样的大小，保持数据的一致性。再然后因为AlexNet, CaffeNet的pre-trained model输入大小在227，所以同样把图片resize到227*227的大小。可能有些样本会出现形变和模糊不清，因为大小不等，没有解决方法。
* normalization
将样本像素值归一化到0~1之间，乘以0.00390625，不要除以255或者256，python2除法只会返回int，不会保留小数。
* zero-centering
减去pixel的均值

#### 训练
在AlexNet, CaffeNet上迁移学习。

#### 预测
测试集同样的处理方法，输出准确度。
