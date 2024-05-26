# NWD
A Normalized Gaussian Wasserstein Distance for Tiny Object Detection

IoU对于微小物体来说不是一个很好的度量，因此用Wasserstein距离度量边界盒相似性的新度量来代替标准IoU。具体来说，首先将边界框建模为二维高斯分布，然后使用提出的归一化Wasserstein距离(NWD)来度量派生高斯分布的相似性。Wasserstein距离的主要优点是，即使没有重叠或重叠可以忽略不计，它也可以度量分布的相似度。此外，NWD对不同尺度的物体不敏感，因此更适合测量微小物体之间的相似性。NWD可以应用于单级和多级锚式探测器。此外，NWD不仅可以代替标签分配中的IoU，还可以代替非最大抑制(NMS)和回归损失函数中的IoU。

添加计算Wasserstein距离的函数，见[nwd.py](https://github.com/ultralytics/yolov5/blob/master/requirements.txt)

```bash
pip install ultralytics
```
