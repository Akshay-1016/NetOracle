# NetOracle

AI-Based Network Attack Forecasting using a Temporal World Model

Smart India Hackathon — Problem Statement 26153  
Organization: National Technical Research Organisation (NTRO)

## Problem

Traditional intrusion detection systems classify individual network flows as benign or malicious. This ignores the temporal structure of an intrusion.

NetOracle instead models a network as an evolving environment and learns how network states transition over time.

The objective is to approximate:

P(S(t+1) | S(t), S(t-1), ...)

and recursively simulate future network states before an attack progresses further.

## Core Architecture

CIC-IDS-2018 Flow Telemetry
        |
        v
10-Second Network State Windows
        |
        v
State Encoder
        |
        v
Causal Temporal Transformer
        |
        v
Latent Network State Z(t)
        |
        v
Learned Transition Dynamics
        |
        +--> Future Network State
        +--> Attack Probability
        +--> MITRE ATT&CK Stage
        |
        v
Recursive K-Step Simulation

NetOracle observes the previous 20 network states:

20 x 10 seconds = 200 seconds of network history

and forecasts the next five states:

5 x 10 seconds = 50-second prediction horizon.

## Dataset

The current prototype uses real CIC-IDS-2018 flow telemetry.

Seven CIC-IDS-2018 traffic days were used during development, covering:

- Benign traffic
- FTP brute force
- SSH brute force
- Web brute force
- XSS
- SQL injection
- Hulk DoS
- SlowHTTPTest
- GoldenEye
- Slowloris
- HOIC DDoS
- LOIC-UDP
- Infiltration

Raw dataset files are not included in this repository due to their size.

Dataset:
https://www.unb.ca/cic/datasets/ids-2018.html

Place CIC-IDS-2018 CSV files under:

data/raw/

## Network State Representation

Every 10-second interval is converted into an 85-dimensional network state.

21 flow-level traffic features are aggregated using:

- Mean
- Standard deviation
- Maximum
- Minimum

An additional flow-count feature is appended.

21 x 4 + 1 = 85 dimensions.

Empty 10-second intervals are explicitly represented rather than removed, preserving temporal continuity.

## World Model

The model contains:

1. Network State Encoder
2. Positional Encoding
3. Causal Temporal Transformer
4. Residual Latent Transition Model
5. Future State Decoder
6. Attack Probability Head
7. MITRE Stage Prediction Head

Unlike a direct multi-output classifier, future latent states are recursively rolled forward:

Z(t)
 -> Z(t+1)
 -> Z(t+2)
 -> ...
 -> Z(t+5)

Each predicted latent state produces:

- Future network state
- Infiltration probability
- MITRE ATT&CK stage

## MITRE ATT&CK Mapping

CIC-IDS-2018 does not provide native MITRE ATT&CK progression labels.

For decision-support demonstration, dataset attack labels are mapped into simplified defensive stages such as:

- Benign
- Initial Access
- Execution / disruptive malicious activity
- Lateral Movement / Infiltration

These mappings are prototype semantic mappings and are not claimed to be official ground-truth MITRE annotations.

## Evaluation Methodology

The model uses chronological separation rather than random flow-level train/test splitting.

Normalization statistics are fitted exclusively on training data.

Validation traffic is used for:

- Model checkpoint selection
- Learning-rate scheduling
- Alert-threshold selection

The final test interval is not used for model fitting or threshold selection.

For the infiltration experiment, an earlier infiltration episode is available during training while a later, temporally separated infiltration episode is used for testing.

## Results

T+1 chronological test results:

World Model:

- F1 Score: 0.694
- AUC-ROC: 0.832
- Precision: 0.620
- Recall: 0.788
- False Positive Rate: 0.344

Logistic Regression baseline:

- F1 Score: 0.583
- AUC-ROC: 0.635
- Precision: 0.486
- Recall: 0.729
- False Positive Rate: 0.548

The World Model therefore provides measurable improvement over the static baseline while learning temporal network dynamics.

The current false-positive rate remains an area for further calibration and enterprise-specific tuning.

## Explainability

NetOracle provides feature attribution using input-gradient attribution.

The dashboard identifies which traffic-state features contribute most strongly to a forecast.

Examples include:

- TCP SYN behavior
- Flow rate statistics
- Packet-size statistics
- Inter-arrival-time characteristics
- Destination port behavior
- Window-level traffic volume

## Dashboard

The Streamlit dashboard provides:

- Real CIC-IDS CSV upload
- Chronological network-state construction
- Attack-risk timeline
- K-step future forecasting
- MITRE ATT&CK stage prediction
- Predictive security alerts
- Defender recommendations
- Feature attribution
- World Model vs Logistic Regression comparison

The application runs fully offline.

## Setup

Recommended Python environment:

Python 3.x with a compatible CUDA-enabled PyTorch installation.

Create a virtual environment:

python -m venv venv

Activate it on Windows Git Bash:

source venv/Scripts/activate

Install dependencies:

pip install -r requirements.txt

Install a CUDA-enabled PyTorch build appropriate for your GPU from:

https://pytorch.org/get-started/locally/

## Training

Place CIC-IDS-2018 CSV files in:

data/raw/

Then run:

python train.py

The best model checkpoint is saved to:

models/best_world_model.pth

Model checkpoints are excluded from Git because of binary size/version portability.

## Running the Dashboard

Run:

python -m streamlit run app/streamlit_app.py

Then open:

http://localhost:8501

## Project Structure

netoracle_project/
|
|-- app/
|   `-- streamlit_app.py
|
|-- configs/
|   `-- config.yaml
|
|-- src/
|   |-- baseline.py
|   |-- explainer.py
|   |-- feature_extraction.py
|   |-- inference.py
|   |-- mitre_mapper.py
|   `-- world_model.py
|
|-- models/
|   |-- evaluation_results.json
|   `-- zero_shot_results.json
|
|-- train.py
|-- requirements.txt
|-- ARCHITECTURE.md
`-- README.md

## Privacy and Deployment

NetOracle is designed for offline execution and does not require cloud APIs.

This makes the architecture applicable to privacy-sensitive environments including enterprise networks and Critical Information Infrastructure.

## Current Prototype Limitations

- Current implementation primarily uses CIC-IDS flow-level telemetry.
- Full PCAP-derived packet-level feature extraction is planned as the next extension.
- MITRE mappings are semantic prototype mappings rather than native dataset annotations.
- False-positive calibration requires environment-specific tuning.
- Cross-dataset validation remains future work.

## Future Work

Planned extensions include:

- PCAP packet-level feature extraction
- Graph-based network-state representation
- Cross-dataset validation
- Calibration and uncertainty estimation
- Real-time NetFlow/IPFIX ingestion
- SOC/SIEM integration
- Automated defensive playbook recommendations