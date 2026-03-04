import torch
import torch.nn as nn
from models.GTShapelet import GTShapelet
from dgl.nn.pytorch import GINConv
from models.utils import get_mask

class GTShapeletNoGNN(nn.Module):
    def __init__(self, k, embed_dim=128, num_heads=4, bert_dim=1536):
        super(GTShapeletNoGNN, self).__init__()
        self.embed_dim = embed_dim
        self.num_nodes = 1 << 2 * k
        self.embed = nn.Embedding(self.num_nodes, self.embed_dim)
        
        # NO GNN Layers
        self.act = nn.GELU()
        
        # Attention Layers
        self.MHA = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.cls_embedding = nn.Parameter(torch.randn([1, 1, embed_dim], requires_grad=True))
        
        # BERT Projection
        self.bert_proj = nn.Linear(bert_dim, embed_dim)
        self.norm_bert = nn.LayerNorm(embed_dim)
        
        self.norm_after = nn.LayerNorm(embed_dim)
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

        # SKIP GNN
        # for gnn in self.convs: ...
                
        # --- Attention Branch ---
        with g.local_scope():
            g.ndata['h'] = h
            src_padding_mask = get_mask(g)
            h = h.view(-1, self.num_nodes, self.embed_dim)
           
        expand_cls_embedding = self.cls_embedding.expand(h.size(0), 1, -1)
        
        tokens_list = [h, expand_cls_embedding]
        
        # If BERT features provided, project and add as a token
        if bert_feat is not None:
             bert_token = self.bert_proj(bert_feat).unsqueeze(1) 
             bert_token = self.norm_bert(bert_token)
             tokens_list.append(bert_token)
             
        # Concatenate and Apply Attention
        h = torch.cat(tokens_list, dim=1)
        
        # Adjust mask for new tokens (CLS + BERT)
        zeros = src_padding_mask.data.new(src_padding_mask.size(0), len(tokens_list) - 1).fill_(0)
        src_padding_mask = torch.cat([src_padding_mask, zeros], dim=1)
        
        attn_output, _ = self.MHA(h, h, h, key_padding_mask=src_padding_mask)
        h = self.norm_after(h + attn_output)
        
        # Return CLS
        if bert_feat is not None:
            return h[:, h.size(1) - 2, :] 
        
        return h[:, -1, :] 

class LncLocNoGNN(nn.Module):
    def __init__(self, k, embed_dim=256, num_heads=4, hidden_dim=256, use_bert=False, bert_dim=1536):
        super(LncLocNoGNN, self).__init__()
        self.use_bert = use_bert
        
        # Use our modified NoGNN encoder
        self.graph_encoder = GTShapeletNoGNN(k, embed_dim, num_heads=num_heads, bert_dim=bert_dim)
        
        input_dim = embed_dim + 200 # Shapelets still included if not explicitly disabled

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
