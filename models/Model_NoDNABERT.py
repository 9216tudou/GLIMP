import torch
import torch.nn as nn
from models.GTShapelet import GTShapelet

class LncLocNoDNABERT(nn.Module):
    def __init__(self, k, embed_dim=256, num_heads=4, hidden_dim=256):
        super(LncLocNoDNABERT, self).__init__()
        # We explicitly set use_bert=False here essentially, 
        # or rather, we don't pass bert_dim or use_bert flag to GTShapelet if checking Model.py
        # But Model.py's GTShapelet init doesn't take use_bert flag, it takes bert_dim.
        # However, GTShapelet forward takes bert_feat.
        
        # Let's double check GTShapelet.py
        # It takes bert_dim in __init__, but if we never pass bert features in forward, 
        # the bert projection layers are just unused weights.
        
        self.graph_encoder = GTShapelet(k, embed_dim, num_heads=num_heads) # bert_dim default is 1536 but unused if input is None
        
        input_dim = embed_dim + 200 # Shapelet included

        self.classify = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, g, sw, sf, bert_embed=None):
        # We ignore bert_embed completely
        sw = torch.nn.functional.normalize(sw, p=2, dim=1)
        
        # Pass None for bert_feat
        graph_repr = self.graph_encoder(g, sw, bert_feat=None)
        
        graph_repr = torch.concat((sf, graph_repr), dim=1)
        
        return self.classify(graph_repr)
