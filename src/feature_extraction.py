"""
Feature Extraction and State Matrix Preparation
"""
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

class FeatureExtractor:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.scaler = StandardScaler()

    def create_synthetic_stream(self, n_windows: int = 3500) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        np.random.seed(42)
        base_features = [
            'dst_port', 'protocol', 'flow_duration', 'total_fwd_packets',
            'total_bwd_packets', 'total_length_fwd', 'total_length_bwd',
            'flow_bytes_per_sec', 'flow_packets_per_sec', 'flow_iat_mean',
            'flow_iat_std', 'flow_iat_max', 'syn_flag_count', 'ack_flag_count',
            'fin_flag_count', 'rst_flag_count', 'psh_flag_count',
            'avg_packet_size', 'fwd_header_length', 'fwd_bwd_ratio',
            'syn_ack_ratio', 'flag_entropy', 'ttl_proxy', 'payload_size_proxy',
            'dst_port_normalized'
        ]
        
        state_dim = len(base_features) * 4 + 1
        states = np.random.randn(n_windows, state_dim).astype(np.float32) * 0.3
        labels = np.zeros(n_windows, dtype=np.int64)
        mitre_stages = np.zeros(n_windows, dtype=np.int64)
        
        # Inject kill-chain attacks
        for _ in range(45):
            start = np.random.randint(15, n_windows - 25)
            chain = [1, 2, 3, 4, 5, 6]
            for offset, stage in enumerate(chain):
                idx = start + offset
                if idx < n_windows:
                    states[idx] += np.random.randn(state_dim) * 0.5 + (stage * 0.8)
                    labels[idx] = 1
                    mitre_stages[idx] = stage
                    
        states = self.scaler.fit_transform(states).astype(np.float32)
        
        seq_len = self.config['data']['sequence_length']
        horizon = self.config['data']['forecast_horizon']
        
        X, y_states, y_attacks, y_mitre = [], [], [], []
        for i in range(len(states) - seq_len - horizon):
            X.append(states[i:i + seq_len])
            y_states.append(states[i + seq_len:i + seq_len + horizon])
            y_attacks.append(labels[i + seq_len:i + seq_len + horizon].astype(np.float32))
            y_mitre.append(mitre_stages[i + seq_len:i + seq_len + horizon])
            
        return (np.array(X, dtype=np.float32),
                np.array(y_states, dtype=np.float32),
                np.array(y_attacks, dtype=np.float32),
                np.array(y_mitre, dtype=np.int64),
                base_features)
