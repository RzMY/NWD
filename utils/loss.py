import torch
import torch.nn as nn
from utils.iou import bbox_iou

def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441"""
    return 1.0 - 0.5 * eps, 0.5 * eps

class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        """Initializes a modified BCEWithLogitsLoss with reduced missing label effects, taking optional alpha smoothing
        parameter.
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """Computes modified BCE loss for YOLOv5 with reduced missing label effects, taking pred and true tensors,
        returns mean loss.
        """
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
        `gamma` and `alpha`.
        """
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

class ComputeLoss:
    # 初始化损失计算类
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False  # 是否对对象的IoU进行排序
        device = next(model.parameters()).device  # 获取模型参数的设备类型
        h = model.hyp  # 获取超参数

        # 定义类别和对象存在性的二元交叉熵损失函数
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # 应用标签平滑
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))

        # 应用焦点损失
        g = h['fl_gamma']
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)


        # m = de_parallel(model).model[-1]  # Detect() module
        det = model.model[-1] if hasattr(model, 'model') else model[-1]  # 获取模型的Detect层
        # 根据层数调整平衡系数
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
        self.ssi = list(det.stride).index(16) if autobalance else 0  # 如果自动平衡，获取步长为16的索引
        # 将初始化的变量赋值给实例
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))
        # self.na = m.na  # number of anchors
        # self.nc = m.nc  # number of classes
        # self.nl = m.nl  # number of layers
        # self.anchors = m.anchors
        # self.device = device

    def __call__(self, p, targets):
        device = targets.device
        # 初始化分类、框和对象存在性损失
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        # 构建目标
        tcls, tbox, indices, anchors = self.build_targets(p, targets)

        # 计算损失
        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)  # 目标对象存在性张量

            n = b.shape[0]  # 目标数量
            if n:
                # 获取与目标对应的预测子集
                ps = pi[b, a, gj, gi]

                # 回归损失
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # 预测的框

                # 计算IoU和Wasserstein距离
                iou = bbox_iou(pbox.T, tbox[i], xywh=True, CIoU=True)
                wasserstein = calculate_nwd(pbox, tbox[i])  # 计算NWD
                # 组合IoU和Wasserstein损失
                lbox += 0.9 * (1.0 - iou).mean() + 0.1 * (1.0 - wasserstein).mean()

                # 对象存在性损失
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # 分配对象存在性分数

                # 分类损失
                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # 目标
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # 二元交叉熵损失

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # 对象存在性损失
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # 批量大小

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # 为compute_loss()构建目标，输入targets格式为(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # 锚点数量，目标数量
        tcls, tbox, indices, anch = [], [], [], []  # 初始化分类目标，框目标，索引，锚点列表
        gain = torch.ones(7, device=targets.device)  # 归一化到网格空间的增益
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # 与.repeat_interleave(nt)相同
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # 添加锚点索引

        g = 0.5  # 偏移量
        off = torch.tensor([[0, 0],  # 定义偏移量，用于微调目标位置
                            [1, 0], [0, 1], [-1, 0], [0, -1]],  # j,k,l,m
                        device=targets.device).float() * g  # 偏移量

        for i in range(self.nl):  # 遍历每个预测层
            anchors = self.anchors[i]  # 当前层的锚点
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy增益

            # 将目标匹配到锚点
            t = targets * gain  # 调整目标到当前层的尺寸
            if nt:
                # 匹配
                r = t[:, :, 4:6] / anchors[:, None]  # 宽高比
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # 比较宽高比，选择最佳匹配的锚点
                t = t[j]  # 筛选

                # 应用偏移量
                gxy = t[:, 2:4]  # 网格xy
                gxi = gain[[2, 3]] - gxy  # 反向偏移
                j, k = ((gxy % 1 < g) & (gxy > 1)).T  # 根据偏移量微调位置
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # 定义
            b, c = t[:, :2].long().T  # 图片编号，类别
            gxy = t[:, 2:4]  # 网格xy
            gwh = t[:, 4:6]  # 网格wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # 网格xy索引

            # 添加到列表
            a = t[:, 6].long()  # 锚点索引
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # 添加图片编号，锚点索引，网格索引
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # 添加框
            anch.append(anchors[a])  # 添加锚点
            tcls.append(c)  # 添加类别

        return tcls, tbox, indices, anch  # 返回分类目标，框目标，索引，锚点

def calculate_nwd(bbox1, bbox2, format_xywh=True): 
    # 定义计算Wasserstein距离的函数，输入两个边界框和格式标志
    bbox2 = bbox2.transpose()  
    # 转置第二个边界框，以匹配第一个边界框的维度
    
    if format_xywh:  # 如果边界框的格式是中心点坐标加宽高
        center_x1, center_y1 = (bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2  
        # 计算第一个边界框的中心点x，y坐标
        width1, height1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]  
        # 计算第一个边界框的宽度和高度
        center_x2, center_y2 = (bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2  
        # 计算第二个边界框的中心点x，y坐标
        width2, height2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]  
        # 计算第二个边界框的宽度和高度
    else:  # 如果边界框的格式是左上角坐标加宽高
        center_x1, center_y1, width1, height1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]  
        # 直接使用第一个边界框的坐标和尺寸
        center_x2, center_y2, width2, height2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]  
        # 直接使用第二个边界框的坐标和尺寸
        
    dist_center_x = (center_x1 - center_x2) ** 2  
    # 计算中心点x坐标差的平方
    dist_center_y = (center_y1 - center_y2) ** 2  
    # 计算中心点y坐标差的平方
    distance_centers = dist_center_x + dist_center_y  
    # 计算中心点之间的距离（欧氏距离的平方）
    
    distance_width = ((width1 - width2) / 2) ** 2  
    # 计算宽度差的一半的平方
    distance_height = ((height1 - height2) / 2) ** 2  
    # 计算高度差的一半的平方
    distance_sizes = distance_width + distance_height  
    # 计算尺寸差异（宽度和高度差异的和）
    
    return distance_centers + distance_sizes  
    # 返回中心点差异和尺寸差异的总和作为Wasserstein距离