import torch
import torch.nn as nn
from models.GTShapelet import GTShapelet


class LncLocNoShapelet(nn.Module):
    def __init__(self, k, embed_dim=256, num_heads=4, hidden_dim=256, use_bert=False, bert_dim=1536):
        super(LncLocNoShapelet, self).__init__()
        self.use_bert = use_bert
        
        # Same graph encoder as main model
        self.graph_encoder = GTShapelet(k, embed_dim, num_heads=num_heads, bert_dim=bert_dim)
        
        # Input dimension does NOT include Shapelet freq (200 dim)
        input_dim = embed_dim 

        self.classify = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, g, sw, sf, bert_embed=None):
        # We accept 'sf' to match the function signature of the training loop, but we ignore it.
        sw = torch.nn.functional.normalize(sw, p=2, dim=1)
        
        effective_bert = None
        if self.use_bert:
            if bert_embed is None:
                raise ValueError("Model initialized with use_bert=True but no bert_embed provided in forward()")
            effective_bert = bert_embed

        graph_repr = self.graph_encoder(g, sw, bert_feat=effective_bert)
        
        # NO concatenation of sf here
        
        return self.classify(graph_repr)
