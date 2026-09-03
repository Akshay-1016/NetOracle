# NetOracle Architecture Document

## 1. System Philosophy
Traditional NIDS perform stateless binary classification on individual packet headers. NetOracle formulates cyber defence as a World Model problem: learning the temporal transition dynamics of an enterprise network state vector over time.

## 2. Model Structure
1. State Encoder: Projects aggregated statistical vectors into a latent representation.
2. Temporal Transformer: Computes multi-head self-attention over historical states with causal masks.
3. Multi-Task Heads:
   - State Simulation Head: Reconstructs future feature vectors.
   - Infiltration Probability Head: Computes attack risk trajectory over horizons.
   - MITRE Stage Head: Classifies alignment with MITRE ATT&CK tactics.

## 3. Explainability
Feature attribution is computed via input-gradient sensitivity, revealing the network flags and ports driving predictions.
