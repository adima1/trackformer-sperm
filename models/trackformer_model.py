import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from collections import deque
import torchvision.models as models


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class TrackFormerModel(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_queries=100, track_memory_size=5):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        )

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)  # adjust to match backbone output

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.track_id_counter = 1
        self.track_memory_size = track_memory_size
        self.track_embeddings_memory = deque(maxlen=track_memory_size)
        self.track_bboxes_memory = deque(maxlen=track_memory_size)
        self.track_ids_memory = deque(maxlen=track_memory_size)
        self.loss_outputs = None

    def forward(self, samples):
        src = self.backbone(samples)
        mask = None
        self.loss_outputs = {
            'loss_track': torch.zeros(1, device=samples.device)
        }

        bs, c, h, w = src.shape
        src = self.input_proj(src).flatten(2).permute(2, 0, 1)  # [HW, B, C]

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # [Q, B, C]
        tgt = torch.zeros_like(query_embed)

        hs = self.transformer(src=src, tgt=tgt)  # [Q, B, C]
        hs = hs.permute(1, 0, 2).unsqueeze(0)    # mimic DETR shape [1, B, Q, C]

        outputs_class = self.class_embed(hs)
        outputs_bbox = self.bbox_embed(hs).sigmoid()

        curr_embeddings = hs[0]  # [B, Q, C]
        curr_bboxes = outputs_bbox[-1]  # [B, Q, 4]
        batch_size = curr_embeddings.shape[0]
        outputs_track_ids = []

        for b in range(batch_size):
            curr_embeds = curr_embeddings[b]
            curr_boxes = curr_bboxes[b]
            track_ids = self.match_to_memory(curr_embeds, curr_boxes)
            outputs_track_ids.append(track_ids)

            if len(self.track_embeddings_memory) > 0:
                prev_embeds = self.track_embeddings_memory[-1]
                prev_ids = self.track_ids_memory[-1]
                loss_track = self.compute_contrastive_loss(curr_embeds, track_ids, prev_embeds, prev_ids)
            else:
                loss_track = torch.tensor(0.0, device=curr_embeds.device)

            if 'loss_track' not in self.loss_outputs:
                self.loss_outputs['loss_track'] = loss_track

            else:
                self.loss_outputs['loss_track'] += loss_track.view(1)

            self.track_embeddings_memory.append(curr_embeds.detach())
            self.track_bboxes_memory.append(curr_boxes.detach())
            self.track_ids_memory.append(track_ids)

        return {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_bbox[-1],
            'pred_track_ids': outputs_track_ids
        }

    def match_to_memory(self, curr_embeds, curr_boxes):
        num_curr = curr_embeds.shape[0]

        if len(self.track_embeddings_memory) == 0:
            ids = torch.arange(self.track_id_counter, self.track_id_counter + num_curr)
            self.track_id_counter += num_curr
            return ids

        prev_embeds = torch.cat(list(self.track_embeddings_memory), dim=0)
        prev_boxes = torch.cat(list(self.track_bboxes_memory), dim=0)
        prev_ids = torch.cat(list(self.track_ids_memory), dim=0)

        curr_norm = F.normalize(curr_embeds, p=2, dim=1)
        prev_norm = F.normalize(prev_embeds, p=2, dim=1)

        similarity_matrix = torch.matmul(curr_norm, prev_norm.T)
        cost_similarity = 1 - similarity_matrix.detach().cpu().numpy()

        def center(box):
            x, y, w, h = box.unbind(-1)
            return torch.stack([x + w / 2, y + h / 2], dim=-1)

        curr_centers = center(curr_boxes)
        prev_centers = center(prev_boxes)
        cost_center = torch.cdist(curr_centers, prev_centers, p=2).detach().cpu().numpy()

        alpha, beta = 0.7, 0.3
        total_cost = alpha * cost_similarity + beta * cost_center

        row_ind, col_ind = linear_sum_assignment(total_cost)
        matched = [-1] * num_curr
        assigned_prev_ids = set()

        for i, j in zip(row_ind, col_ind):
            if total_cost[i, j] < 0.6:
                matched[i] = prev_ids[j].item()
                assigned_prev_ids.add(prev_ids[j].item())

        for i in range(num_curr):
            if matched[i] == -1:
                matched[i] = self.track_id_counter
                self.track_id_counter += 1

        return torch.tensor(matched, dtype=torch.int64)

    def compute_contrastive_loss(self, curr_embeds, curr_ids, prev_embeds, prev_ids, margin=1.0):
        loss = 0.0
        count = 0
        for i in range(len(curr_ids)):
            for j in range(len(prev_ids)):
                same = curr_ids[i] == prev_ids[j]
                dist = F.pairwise_distance(curr_embeds[i].unsqueeze(0), prev_embeds[j].unsqueeze(0), p=2)
                if same:
                    loss += dist ** 2
                else:
                    loss += torch.clamp(margin - dist, min=0) ** 2
                count += 1
        return loss / max(count, 1)
