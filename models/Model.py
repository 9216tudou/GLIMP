import torch
import torch.nn as nn
from models.GTShapelet import GTShapelet


class LncLoc(nn.Module):
    def __init__(self, k, embed_dim=256, num_heads=4, hidden_dim=256, use_bert=False, bert_dim=1536):
        super(LncLoc, self).__init__()
        self.use_bert = use_bert
        
        # Revert to original signature but keep bert support
        self.graph_encoder = GTShapelet(k, embed_dim, num_heads=num_heads, bert_dim=bert_dim)
        
        # Original logic: input_dim is embed_dim + 200 (shapelet)
        # If BERT is used, it is fused in GTShapelet so dimensions are handled there.
        # But wait, did original LncLoc hardcode + 200? Yes.
        
        input_dim = embed_dim + 200

        self.classify = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, g, sw, sf, bert_embed=None):
        sw = torch.nn.functional.normalize(sw, p=2, dim=1)
        
        # If use_bert is False, we ignore bert_embed even if passed
        effective_bert = None
        if self.use_bert:
            if bert_embed is None:
                raise ValueError("Model initialized with use_bert=True but no bert_embed provided in forward()")
            effective_bert = bert_embed

        graph_repr = self.graph_encoder(g, sw, bert_feat=effective_bert)
        
        # Concatenate Shapelet features (sf)
        graph_repr = torch.concat((sf, graph_repr), dim=1)
        
        return self.classify(graph_repr)

    def load_pretrain_encoder(self, source):
        source_state = source
        for name, param in self.named_parameters():
            if name in source_state:
                param.data = source_state[name]

    def set_parameter(self):
        for name, param in self.named_parameters():
            if 'graph_encoder' in name:
                continue
            if 'norm' in name:
                continue
            if 'bias' in name:
                nn.init.zeros_(param)
                continue
            nn.init.kaiming_uniform_(param)
