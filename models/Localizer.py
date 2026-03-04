import pickle

import pandas as pd
import torch

from models.Model import LncLoc
from models.utils import get_neighbor, idng, generate_weight_by_shapelet


class Localizer:
    def __init__(self, weightPath, shapePath, k=3, embed_dim=256,
                 hidden_dim=256, window_size=40, device='cuda:0'):
        self.k = k
        self.device = device
        self.window_size = window_size
        model = LncLoc(k=k, embed_dim=embed_dim, hidden_dim=hidden_dim)
        state = torch.load(weightPath, map_location=torch.device(device))
        state.pop('norm.weight')
        state.pop('norm.bias')
        model.load_state_dict(state)
        model.to(device)
        self.model = model
        self.neighbor_dict = get_neighbor(k)
        with open(shapePath, 'rb') as f:
            self.shapelet_info=pickle.load(f)

        self.sigmoid = torch.nn.Sigmoid()

    def predict(self, RNA):
        self.model.eval()
        graph = idng(RNA, self.k, self.neighbor_dict, self.window_size)
        sw = generate_weight_by_shapelet(RNA, self.shapelet_info, self.k)
        sw = torch.FloatTensor(sw)
        sf = torch.FloatTensor([RNA.count(item[0]) for item in self.shapelet_info])
        score = self.model(graph.to(self.device), sw.view(1, -1).to(self.device), sf.view(1, -1).to(self.device))
        score = self.sigmoid(score)
        return score.detach().cpu().numpy()
