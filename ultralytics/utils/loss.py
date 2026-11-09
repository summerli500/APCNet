# Detect loss: BCE + DFL + Inner-Wise-CIoU.

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

from .metrics import WiseIouLoss
from .tal import bbox2dist


def _env(key, default=""):
    return os.environ.get(key, default)


def _env_float(key, default):
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return float(default)


class DFLoss(nn.Module):
    """Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max=16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Box regression: Inner-Wise-CIoU (+ DFL)."""

    def __init__(self, reg_max=16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.iou_mode = (_env("YOLO_BBOX_IOU", "wise_inner_ciou") or "wise_inner_ciou").lower()
        if self.iou_mode != "wise_inner_ciou":
            raise ValueError(f"YOLO_BBOX_IOU must be wise_inner_ciou, got {self.iou_mode!r}")
        self.inner_ratio = _env_float("YOLO_INNER_RATIO", 0.7)
        self.wiou_loss = WiseIouLoss(
            ltype="CIoU",
            monotonous=False,
            inner_iou=True,
            focaler_iou=False,
        )

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        mpdiou_hw=None,
    ):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pb, tb = pred_bboxes[fg_mask], target_bboxes[fg_mask]
        wiou = self.wiou_loss(
            pb, tb, ret_iou=False, ratio=self.inner_ratio, d=0.0, u=0.95
        ).unsqueeze(-1)
        loss_iou = (wiou * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class v8DetectionLoss:
    """Detect training criterion."""

    def __init__(self, model, tal_topk=10):
        device = next(model.parameters()).device
        h = model.args
        m = model.model[-1]
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device
        self.use_dfl = m.reg_max > 1
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        if hasattr(m, "dfl_aux"):
            self.assigner_aux = TaskAlignedAssigner(topk=13, num_classes=self.nc, alpha=0.5, beta=6.0)
            self.aux_loss_ratio = 0.25
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        if hasattr(self, "assigner_aux"):
            loss, batch_size = self.compute_loss_aux(preds, batch)
        else:
            loss, batch_size = self.compute_loss(preds, batch)
        return loss.sum() * batch_size, loss.detach()

    def compute_loss(self, preds, batch):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        feats = feats[: self.stride.size(0)]
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                ((imgsz[0] ** 2 + imgsz[1] ** 2) / torch.square(stride_tensor))
                .repeat(1, batch_size)
                .transpose(1, 0),
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss, batch_size

    def compute_loss_aux(self, preds, batch):
        """Auxiliary DFL heads (kept for compatibility)."""
        loss = torch.zeros(3, device=self.device)
        feats_all = preds[1] if isinstance(preds, tuple) else preds
        if len(feats_all) == self.stride.size(0):
            return self.compute_loss(preds, batch)
        feats, feats_aux = feats_all[: self.stride.size(0)], feats_all[self.stride.size(0) :]

        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        pred_distri_aux, pred_scores_aux = torch.cat(
            [xi.view(feats_aux[0].shape[0], self.no, -1) for xi in feats_aux], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores, pred_distri = (
            pred_scores.permute(0, 2, 1).contiguous(),
            pred_distri.permute(0, 2, 1).contiguous(),
        )
        pred_scores_aux, pred_distri_aux = (
            pred_scores_aux.permute(0, 2, 1).contiguous(),
            pred_distri_aux.permute(0, 2, 1).contiguous(),
        )

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        pred_bboxes_aux = self.bbox_decode(anchor_points, pred_distri_aux)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        _, target_bboxes_aux, target_scores_aux, fg_mask_aux, _ = self.assigner_aux(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        target_scores_sum_aux = max(target_scores_aux.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        loss[1] += (
            self.bce(pred_scores_aux, target_scores_aux.to(dtype)).sum()
            / target_scores_sum_aux
            * self.aux_loss_ratio
        )

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            target_bboxes_aux /= stride_tensor
            mpdiou_hw = (
                ((imgsz[0] ** 2 + imgsz[1] ** 2) / torch.square(stride_tensor))
                .repeat(1, batch_size)
                .transpose(1, 0)
            )
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                mpdiou_hw,
            )
            aux_loss_0, aux_loss_2 = self.bbox_loss(
                pred_distri_aux,
                pred_bboxes_aux,
                anchor_points,
                target_bboxes_aux,
                target_scores_aux,
                target_scores_sum_aux,
                fg_mask_aux,
                mpdiou_hw,
            )
            loss[0] += aux_loss_0 * self.aux_loss_ratio
            loss[2] += aux_loss_2 * self.aux_loss_ratio

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss, batch_size
