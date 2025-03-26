import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=1.0, cost_giou=1.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs['pred_logits'].shape[:2]

        out_prob = outputs['pred_logits'].softmax(-1)  # [B, Q, num_classes]
        out_bbox = outputs['pred_boxes']              # [B, Q, 4]

        indices = []
        print(f"[DEBUG] bs={bs}, len(targets)={len(targets)}")
        for b in range(bs):
            tgt_ids = targets[b]['labels']
            tgt_bbox = targets[b]['boxes']

            cost_class = -out_prob[b][:, tgt_ids]  # [Q, num_targets]
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)
            cost = self.cost_class * cost_class + self.cost_bbox * cost_bbox

            cost = cost.cpu()
            indices_b = linear_sum_assignment(cost)
            indices.append((torch.as_tensor(indices_b[0], dtype=torch.int64),
                            torch.as_tensor(indices_b[1], dtype=torch.int64)))

        return indices

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        return F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=self.empty_weight)

    def loss_boxes(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        return F.l1_loss(src_boxes, target_boxes, reduction='none').sum() / src_boxes.size(0)

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        loss_cls = self.loss_labels(outputs, targets, indices)
        loss_bbox = self.loss_boxes(outputs, targets, indices)
        return loss_cls + loss_bbox
