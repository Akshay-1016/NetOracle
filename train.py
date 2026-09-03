"""
Training and Evaluation Pipeline for NetOracle World Model
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import yaml

from src.feature_extraction import FeatureExtractor
from src.world_model import NetworkWorldModel
from src.baseline import BaselineModel

def run():
    print("="*60)
    print("NETORACLE: WORLD MODEL TRAINING & BENCHMARKING")
    print("="*60)
    
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    extractor = FeatureExtractor("configs/config.yaml")
    X, y_states, y_attacks, y_mitre, feature_names = extractor.create_synthetic_stream()
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    ys_train, ys_test = y_states[:split], y_states[split:]
    ya_train, ya_test = y_attacks[:split], y_attacks[split:]
    ym_train, ym_test = y_mitre[:split], y_mitre[split:]
    
    train_loader = DataLoader(TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(ys_train),
        torch.FloatTensor(ya_train), torch.LongTensor(ym_train)
    ), batch_size=config['model']['batch_size'], shuffle=True)
    
    test_loader = DataLoader(TensorDataset(
        torch.FloatTensor(X_test), torch.FloatTensor(ys_test),
        torch.FloatTensor(ya_test), torch.LongTensor(ym_test)
    ), batch_size=config['model']['batch_size'], shuffle=False)
    
    input_dim = X.shape[-1]
    model = NetworkWorldModel(
        input_dim=input_dim,
        state_dim=config['model']['state_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        forecast_horizon=config['data']['forecast_horizon']
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['model']['learning_rate'], weight_decay=1e-4)
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCELoss()
    ce_fn = nn.CrossEntropyLoss()
    
    print("[*] Training Temporal World Model Dynamics...")
    for epoch in range(config['model']['epochs']):
        model.train()
        total_loss = 0
        for bx, bys, bya, bym in train_loader:
            bx, bys, bya, bym = bx.to(device), bys.to(device), bya.to(device), bym.to(device)
            optimizer.zero_grad()
            out = model(bx)
            
            l_state = mse_fn(out['predicted_states'], bys)
            l_attack = bce_fn(out['attack_probs'], bya)
            l_stage = ce_fn(out['mitre_logits'].reshape(-1, 7), bym.reshape(-1))
            
            loss = l_state + (2.0 * l_attack) + (1.5 * l_stage)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:2d}/{config['model']['epochs']}] - Loss: {total_loss/len(train_loader):.4f}")
            
    # Evaluation
    model.eval()
    all_preds, all_trues, all_probs = [], [], []
    with torch.no_grad():
        for bx, bys, bya, bym in test_loader:
            bx = bx.to(device)
            out = model(bx)
            probs = out['attack_probs'][:, 0].cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend((probs > 0.5).astype(int))
            all_trues.extend((bya[:, 0] > 0.5).numpy().astype(int))
            
    wm_metrics = {
        'accuracy': float(accuracy_score(all_trues, all_preds)),
        'f1_score': float(f1_score(all_trues, all_preds, zero_division=0)),
        'precision': float(precision_score(all_trues, all_preds, zero_division=0)),
        'recall': float(recall_score(all_trues, all_preds, zero_division=0)),
        'auc_roc': float(roc_auc_score(all_trues, all_probs))
    }
    
    print("\n[*] Evaluating Static Baseline (Logistic Regression)...")
    baseline = BaselineModel()
    bl_metrics = baseline.evaluate(X, y_attacks)
    
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'config': config,
        'input_dim': input_dim,
        'feature_names': feature_names
    }, 'models/best_world_model.pth')
    
    results = {
        'world_model': wm_metrics,
        'baseline': bl_metrics,
        'f1_gain_percent': ((wm_metrics['f1_score'] - bl_metrics['f1_score']) / (bl_metrics['f1_score'] + 1e-8)) * 100
    }
    
    with open('models/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*60)
    print("BENCHMARK COMPARISON RESULTS")
    print("="*60)
    print(f"World Model F1: {wm_metrics['f1_score']:.4f} | Baseline F1: {bl_metrics['f1_score']:.4f}")
    print(f"Improvement: +{results['f1_gain_percent']:.2f}% over static classification")
    print("[+] Model checkpoint saved -> models/best_world_model.pth")
    print("[+] Results saved -> models/evaluation_results.json")

if __name__ == '__main__':
    run()
