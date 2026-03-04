from dgl.nn.pytorch import GINConv
import torch.nn as nn
import torch
from models.utils import get_mask


class GTShapelet(nn.Module):
    def __init__(self, k, embed_dim=128, num_heads=4, bert_dim=1536):
        super(GTShapelet, self).__init__()
        self.embed_dim = embed_dim
        self.num_nodes = 1 << 2 * k
        self.embed = nn.Embedding(self.num_nodes, self.embed_dim)
        
        # GNN Layers
        self.convs = nn.ModuleList()
        self.convs.append(GINConv(nn.Linear(embed_dim, 2 * embed_dim)))
        self.convs.append(GINConv(nn.Linear(2 * embed_dim, 2 * embed_dim)))
        self.convs.append(GINConv(nn.Linear(2 * embed_dim, embed_dim)))
            
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

        for gnn in self.convs:
            h = gnn(g, h, edge_weight=g.edata['weight'].float())
            h = self.act(h)
                
        # --- Attention Branch ---
        with g.local_scope():
            g.ndata['h'] = h
            src_padding_mask = get_mask(g)
            h = h.view(-1, self.num_nodes, self.embed_dim)  # batch_size  * num_node * embed
           # h = torch.einsum('bne,bn->bne', h, sw)
           
        expand_cls_embedding = self.cls_embedding.expand(h.size(0), 1, -1)  # batchsize * 1 * embed
        
        tokens_list = [h, expand_cls_embedding]
        
        # If BERT features provided, project and add as a token
        if bert_feat is not None:
             bert_token = self.bert_proj(bert_feat).unsqueeze(1) # [Batch, 1, Embed]
             bert_token = self.norm_bert(bert_token)
             tokens_list.append(bert_token)
             
        h = torch.cat(tokens_list, dim=1)  # batch * (length + 1 or 2) * dim
        
        # Adjust mask for new tokens
        # Original mask covers 'h' (num_nodes). CLS and BERT are valid (0 padding)
        zeros = src_padding_mask.data.new(src_padding_mask.size(0), len(tokens_list) - 1).fill_(0)
        src_padding_mask = torch.cat([src_padding_mask, zeros], dim=1)
        
        attn_output, _ = self.MHA(h, h, h, key_padding_mask=src_padding_mask)
        h = self.norm_after(h + attn_output)
        
        # Return CLS
        if bert_feat is not None:
            return h[:, h.size(1) - 2, :] # Return CLS (second to last)
        
        return h[:, -1, :] # Return CLS (last)
