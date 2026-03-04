import torch
import numpy as np
import os
import sys
import logging
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
import warnings
import pandas as pd

# Import from existing project files
from config import config
from models.Model import LncLoc
from data.idng_dataset import idng_dataset
from shaplet import get_shapelet
from models.utils import make_path, plot_AUROC, plot_training_curves
from generate_bert_features import generate_bert_features

# Import training utilities from main.py
try:
    from main import train_step, test_performance
except ImportError:
    # If main.py imports fail, redefine simple versions or ensure path is correct
    sys.path.append(os.getcwd())
    from main import train_step, test_performance

warnings.filterwarnings("ignore")

def train_real_model():
    # 1. Configuration
    conf = config()
    conf.epoch = 30 
    conf.data_path = 'data/human/nonTATA/promoter300/train_keep_replace.csv' # Target dataset (Combined Real Neg + Original Pos)
    conf.res_dir = 'finalResult/real_model' # Output directory
    conf.batch_size = 64 # Smaller batch size for better generalization sometimes
    
    make_path(conf.res_dir)
    
    # Set seed
    torch.manual_seed(conf.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(conf.seed)
        conf.device = 'cuda:0'
    else:
        conf.device = 'cpu'
    
    print(f"--- 开始真实场景训练 ---")
    print(f"数据集: {conf.data_path}")
    print(f"输出目录: {conf.res_dir}")
    
    # 2. Recalculate Shapelets based on the NEW training data
    print("正在从真实训练数据计算 shapelets（这可能需要一些时间）...")
    if conf.use_shapelet:
        try:
            # We explicitly pass the new training path to get_shapelet
            shapelet_info = get_shapelet(path=conf.data_path, k_=4)[:200]
            print(f"已计算 {len(shapelet_info)} 个 shapelets。")
        except Exception as e:
            print(f"计算 shapelets 时出错: {e}")
            shapelet_info = []
    else:
        shapelet_info = []

    # 3. Load Dataset
    print("正在加载图数据集...")
    # Using a unique cache path
    cache_path = "ckpt/real_train_cache"
    
    # Pre-compute BERT features
    bert_path = generate_bert_features(conf.data_path)

    try:
        dataset = idng_dataset(
            data_path=conf.data_path,
            k=conf.k,
            shapelet_info=shapelet_info,
            data_save_path=cache_path, 
            window_size=conf.window_size,
            reload=True, # Force reload to ensure new graphs are built
            bert_path=bert_path
        )
    except FileNotFoundError:
        print(f"错误: 找不到 {conf.data_path}。请先运行 build_generalization_dataset.py。")
        return

    # 4. Training Loop (5-Fold)
    kf = KFold(n_splits=5, shuffle=True, random_state=conf.seed)
    
    best_global_auc = 0
    
    # We will save the shapelets too, so inference script can use them
    import pickle
    with open(os.path.join(conf.res_dir, 'shapelets.pkl'), 'wb') as f:
        pickle.dump(shapelet_info, f)
    print(f"已保存 shapelets 至 {os.path.join(conf.res_dir, 'shapelets.pkl')}")

    for fold_idx, (train_idx, dev_idx) in enumerate(kf.split(dataset)):
        print(f"\n=== 第 {fold_idx + 1}/5 折 ===")
        
        train_sampler = SubsetRandomSampler(train_idx)
        dev_sampler = SubsetRandomSampler(dev_idx)
        
        # Sampler in DGL GraphDataLoader
        train_loader = GraphDataLoader(dataset, batch_size=conf.batch_size, sampler=train_sampler)
        val_loader = GraphDataLoader(dataset, batch_size=conf.batch_size, sampler=dev_sampler)

        model = LncLoc(k=conf.k, embed_dim=conf.embed_dim, hidden_dim=conf.hidden_dim, use_bert=True)
        model = model.to(conf.device)
        
        opt = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=conf.decay)
        
        history = {"epoch": [], "loss": [], "auroc": [], "auprc": [], "acc": [], "mcc": [], "f1": []}
        
        for epoch in range(conf.epoch):
            # Train
            try:
                epoch_loss = train_step(train_loader, model, opt, conf)
                
                # Convert loss to float if tensor
                if isinstance(epoch_loss, torch.Tensor):
                    epoch_loss = epoch_loss.item()
                
                # Validate
                val_perf = test_performance(val_loader, model, threshold=0.5, config=conf)
                
                print(f"Epoch {epoch+1}/{conf.epoch} | Loss: {epoch_loss:.4f} | Val AUROC: {val_perf['auroc']:.4f}")
                
                # History record
                history["epoch"].append(epoch + 1)
                history["loss"].append(epoch_loss)
                history["auroc"].append(val_perf["auroc"])
                history["auprc"].append(val_perf["auprc"])
                history["acc"].append(val_perf["accuracy"])
                history["mcc"].append(val_perf["mcc"])
                history["f1"].append(val_perf["f1"])

                # Checkpoint
                if val_perf["auroc"] > best_global_auc:
                    best_global_auc = val_perf["auroc"]
                    
                    # Save model
                    torch.save(model.state_dict(), f"{conf.res_dir}/best.pt")
                    print(f"--> [新纪录] 已保存模型，AUROC: {best_global_auc:.4f}")
            
            except Exception as e:
                print(f"训练步骤出错: {e}")
                import traceback
                traceback.print_exc()
                break
                
        # 绘制本折的训练曲线
        plot_training_curves(history, conf.res_dir)

    print(f"\n训练完成。最佳验证 AUROC: {best_global_auc:.4f}")
    print(f"模型已保存至: {conf.res_dir}/best.pt")
    print(f"Shapelets 已保存至: {conf.res_dir}/shapelets.pkl")
    print("您现在可以使用 test_generalization.py 运行测试。")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Allow overriding data path
        # python train_real.py data/real_train.csv
        from config import config # re-import to modify instance if needed, but we used local conf
        # Actually simplest to just modify the default path in the code or use argparse
        pass
        
    train_real_model()
