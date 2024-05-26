# NWD
A Normalized Gaussian Wasserstein Distance for Tiny Object Detection [论文链接](https://arxiv.org/abs/2110.13389)

IoU对于微小物体来说不是一个很好的度量，因此用Wasserstein距离度量边界盒相似性的新度量来代替标准IoU。具体来说，首先将边界框建模为二维高斯分布，然后使用提出的归一化Wasserstein距离(NWD)来度量派生高斯分布的相似性。Wasserstein距离的主要优点是，即使没有重叠或重叠可以忽略不计，它也可以度量分布的相似度。此外，NWD对不同尺度的物体不敏感，因此更适合测量微小物体之间的相似性。NWD可以应用于单级和多级锚式探测器。此外，NWD不仅可以代替标签分配中的IoU，还可以代替非最大抑制(NMS)和回归损失函数中的IoU。

# 不通用使用方法(YOLO)
1.添加计算Wasserstein距离的函数calculate_nwd，见[nwd.py](https://github.com/RzMY/NWD/blob/main/utils/nwd.py)

2.在class ComputeLoss:中的def __call__函数中修改：
```bash
# 计算IoU和Wasserstein距离
iou = bbox_iou(pbox.T, tbox[i], xywh=True, CIoU=True)
wasserstein = calculate_nwd(pbox, tbox[i])  # 计算NWD
# 组合IoU和Wasserstein损失
lbox += 0.9 * (1.0 - iou).mean() + 0.1 * (1.0 - wasserstein).mean()
```
