import os

import torch
from torch.utils.data import DataLoader
from dgl.dataloading import GraphDataLoader
import numpy as np
import pandas as pd
from config import config
from models.Model import LncLoc
from data.idng_dataset import idng_dataset
from models.utils import evaluate_performance
from shaplet import get_shapelet
import pickle
import sys
from generate_bert_features import generate_bert_features

def test_generalization(test_dataset_path, model_path=None, shapelet_train_csv=None):
    """
    Tests the model on a provided dataset (containing both positives and negatives).
    """
    conf = config()
    
    # LOAD SHAPELETS
    # Priority 1: User specified CSV path (re-calculate)
    # Priority 2: Pickle file in same directory as model_path
    # Priority 3: Default pickle file
    # Priority 4: Default calculation
    
    shapelet_info = []
    
    # Check for shapelets.pkl in model directory
    model_dir_shapelets = os.path.join(os.path.dirname(model_path), "shapelets.pkl") if model_path else None
    
    default_pickle_path = "finalResult/train_keep_replace_k3_idng.pkl"
    
    if shapelet_train_csv:
        print(f"Calculating shapelets from provided source: {shapelet_train_csv}...")
        shapelet_info = get_shapelet(path=shapelet_train_csv, k_=4)[:200] if conf.use_shapelet else []
        
    elif model_dir_shapelets and os.path.exists(model_dir_shapelets):
        print(f"Loading shapelets from model directory: {model_dir_shapelets}...")
        with open(model_dir_shapelets, 'rb') as f:
            shapelet_info = pickle.load(f)

    elif os.path.exists(default_pickle_path) and conf.use_shapelet:
        print(f"Loading shapelets from default pickle: {default_pickle_path}...")
        with open(default_pickle_path, 'rb') as f:
            shapelet_info = pickle.load(f)
            if not isinstance(shapelet_info, list):
                print(f"Warning: Loaded shapelet info is type {type(shapelet_info)}")
    else:
        print("Warning: No shapelet source provided and default pickle not found. Using default training data path.")
        shapelet_info = get_shapelet(k_=4)[:200] if conf.use_shapelet else []
    
    print(f"Loading Test Dataset from {test_dataset_path}...")
    
    # Pre-compute BERT features
    bert_features_path = generate_bert_features(test_dataset_path)

    # Load and process dataset
    dataset = idng_dataset(
        data_path=test_dataset_path,
        k=conf.k,
        data_save_path=f"ckpt/generalization_test",
        shapelet_info=shapelet_info,
        reload=True,
        window_size=conf.window_size,
        bert_path=bert_features_path
    )
    
    dataloader = GraphDataLoader(dataset, batch_size=conf.batch_size, shuffle=False)
    
    # Load Model
    # Assumes model was trained with use_bert=True if we are here
    model = LncLoc(k=conf.k, embed_dim=conf.embed_dim, hidden_dim=conf.hidden_dim, use_bert=True)
    model = model.to(conf.device)
    
    if model_path:
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Warning: No model checkpoint provided. Model is random initialized!")

    model.eval()
    prob = []
    target = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 5:
                graph, label, sw, sf, bert_embed = batch
            elif len(batch) == 4:
                # Fallback if somehow dataset loads 4, but we init with bert=True?
                # This should not happen if bert_path is passed.
                # But if bert generation fails, dataset might not return it.
                graph, label, sw, sf = batch
                bert_embed = None
                
            graph = graph.to(conf.device)
            sw = sw.to(conf.device)
            sf = sf.to(conf.device)
            
            if bert_embed is not None:
                bert_embed = bert_embed.to(conf.device)
                logit = model(graph, sw, sf, bert_embed)
            else:
                 # Should fail if use_bert=True and no embed
                 logit = model(graph, sw, sf)

            label = label.float().unsqueeze(-1).to(conf.device)
            
            probs_batch = torch.sigmoid(logit).cpu().numpy().flatten()
            targets_batch = label.cpu().numpy().flatten()
            
            prob.extend(probs_batch)
            target.extend(targets_batch)
            
    prob = np.array(prob)
    target = np.array(target)

    # Debug Statistics
    print("\n--- Diagnostic Statistics ---")
    print(f"Total samples: {len(prob)}")
    print(f"Positive samples: {np.sum(target==1)}")
    print(f"Negative samples: {np.sum(target==0)}")
    print(f"Mean Prediction Probability: {np.mean(prob):.4f}")
    print(f"Median Prediction Probability: {np.median(prob):.4f}")
    if np.sum(target==1) > 0:
        print(f"Mean Prob for Positives: {np.mean(prob[target==1]):.4f}")
    if np.sum(target==0) > 0:
        print(f"Mean Prob for Negatives: {np.mean(prob[target==0]):.4f}")
    print("-----------------------------\n")
            
    metrics = evaluate_performance(target, (prob > 0.5).astype(int), prob)
    print("\nGeneralization Performance:")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"AUPRC: {metrics['auprc']:.4f}")
    print(f"ACC:   {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    # Example usage:
    # python test_generalization.py data/real_generalization_dataset.csv finalResult/best.pt [shapelet_train_csv]
    #python test_generalization.py data/real_generalization_dataset.csv finalResult/best.pt data/real_generalization_dataset.csv
    
    if len(sys.argv) < 3:
        print("Usage: python test_generalization.py <test_dataset_csv> <model_ckpt> [shapelet_train_csv]")
    else:
        test_csv = sys.argv[1]
        ckpt = sys.argv[2]
        shapelet_csv = sys.argv[3] if len(sys.argv) > 3 else None
        
        try:
            test_generalization(test_csv, ckpt, shapelet_csv)
        except Exception as e:
            print(f"Error during testing: {e}")
            import traceback
            traceback.print_exc()
