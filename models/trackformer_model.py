import torch
import torch.nn as nn
from backbone.resnet_backbone import Backbone, PositionEmbeddingSine
from transformer.transformer_module import Transformer

class TrackFormerModel(nn.Module):
    def __init__(self, num_classes=1, hidden_dim=256, num_queries=100):
        super(TrackFormerModel, self).__init__()

        # Backbone - ResNet50
        self.backbone = Backbone(name='resnet50')
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)

        # Positional Encoding
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=hidden_dim // 2, normalize=True)

        # Transformer
        self.transformer = Transformer(d_model=hidden_dim, nhead=8, num_encoder_layers=6, num_decoder_layers=6)

        # Object Detection Head - class + bbox
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # כולל רקע
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

        # Tracking Head - הפקת embedding לעקיבה
        self.track_embed = nn.Linear(hidden_dim, hidden_dim)

        # Query Embedding
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

    def forward(self, samples):
        """
        samples: Tensor of shape [B, 3, H, W]
        """
        # שלב 1: הפקת פיצ'רים מה-backbone
        features = self.backbone(samples)  # מחזיר dict עם {'0': Tensor}
        src = self.input_proj(features['0'])  # מקרין מ-2048 ל-256


        # שלב 2: positional encoding
        pos = self.position_encoding(src)  # [B, hidden_dim, H, W]

        # שלב 3: יצירת מסכה (כרגע בלי אזורים חסומים)
        B, C, H, W = src.shape
        mask = torch.zeros((B, H, W), dtype=torch.bool, device=src.device)

        # שלב 4: Transformer
        hs, memory = self.transformer(src, mask, self.query_embed.weight, pos)
        print("hs shape:", hs.shape)

        # שלב 5: Object Detection Head
        outputs_class = self.class_embed(hs)        # [num_layers, B, num_queries, num_classes + 1]
        outputs_coord = self.bbox_embed(hs).sigmoid()  # [num_layers, B, num_queries, 4] – normalized bbox

        # שלב 6: Tracking Head - יוצא embedding לכל query לעקיבה
        outputs_track = self.track_embed(hs)         # [num_layers, B, num_queries, hidden_dim]

        return {
            'pred_logits': outputs_class[-1],   # התחזיות למחלקות
            'pred_boxes': outputs_coord[-1],    # התיבות החזויות
            'track_embeddings': outputs_track[-1]  # embedding לעקיבה
        }
