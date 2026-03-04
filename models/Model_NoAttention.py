import torch
import torch.nn as nn
from models.GTShapelet import GTShapelet
from dgl.nn.pytorch import GINConv
from models.utils import get_mask

class GTShapeletNoAttention(nn.Module):
    def __init__(self, k, embed_dim=128, num_heads=4, bert_dim=1536):
        super(GTShapeletNoAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_nodes = 1 << 2 * k
        self.embed = nn.Embedding(self.num_nodes, self.embed_dim)
        
        # GNN Layers
        self.convs = nn.ModuleList()
        self.convs.append(GINConv(nn.Linear(embed_dim, 2 * embed_dim)))
        self.convs.append(GINConv(nn.Linear(2 * embed_dim, 2 * embed_dim)))
        self.convs.append(GINConv(nn.Linear(2 * embed_dim, embed_dim)))
            
        self.act = nn.GELU()
        
        # NO Attention Layers
        # We will use Mean Pooling
        
        # BERT Projection
        self.bert_proj = nn.Linear(bert_dim, embed_dim)
        self.norm_bert = nn.LayerNorm(embed_dim)
        
        self.set_parameter()

    def set_parameter(self):
        for name, param in self.named_parameters():
            if 'norm' in name:
                continue
            if 'bias' in name:
                nn.init.zeros_(param)
                continue
            nn.init.kaiming_uniform_(param)

    def forward(self, g, sw, bert_feat=None):
        h = self.embed(g.ndata['mask'])

        for gnn in self.convs:
            h = gnn(g, h, edge_weight=g.edata['weight'].float())
            h = self.act(h)
                
        # --- Mean Pooling Branch (No Attention) ---
        with g.local_scope():
            g.ndata['h'] = h
            # Reshape to [Batch, NumNodes, Dim] for pooling
            h = h.view(-1, self.num_nodes, self.embed_dim)
            
            # Simple average of node embeddings
            final_repr = torch.mean(h, dim=1)
            
            # If BERT is present, fuse it 
            if bert_feat is not None:
                bert_encoded = self.bert_proj(bert_feat)
                bert_encoded = self.norm_bert(bert_encoded)
                # Average graph representation and BERT representation
                # Or concat? Original LncLoc concatenated BERT at the end or integrated via Attn.
                # Since we want to maintain dimensions for classifier compatibility, averaging is safer if dim is same.
                # Let's average to simulate "fusion without attention"
                final_repr = (final_repr + bert_encoded) / 2
                
            return final_repr

class LncLocNoAttention(nn.Module):
    def __init__(self, k, embed_dim=256, num_heads=4, hidden_dim=256, use_bert=False, bert_dim=1536):
        super(LncLocNoAttention, self).__init__()
        self.use_bert = use_bert
        
        # Use our modified NoAttention encoder
        self.graph_encoder = GTShapeletNoAttention(k, embed_dim, num_heads=num_heads, bert_dim=bert_dim)
        
        # Input dimension includes Shapelet freq (200 dim)
        input_dim = embed_dim + 200

        self.classify = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, g, sw, sf, bert_embed=None):
        sw = torch.nn.functional.normalize(sw, p=2, dim=1)
        
        effective_bert = None
        if self.use_bert:
            if bert_embed is None:
                raise ValueError("Model initialized with use_bert=True but no bert_embed provided in forward()")
            effective_bert = bert_embed

        graph_repr = self.graph_encoder(g, sw, bert_feat=effective_bert)
        graph_repr = torch.concat((sf, graph_repr), dim=1)
        
        return self.classify(graph_repr)
