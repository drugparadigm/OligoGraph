# **OliGraphs: A Graph Attention Framework for Rational siRNA Efficacy Prediction**

OliGraphs is a multi-modal deep learning framework designed to capture the complex interplay between sequence, structure, and thermodynamics in siRNA efficacy prediction.

The architecture initializes by fusing pre-trained foundation model embeddings with one-hot encoded nucleotide sequences, which are then processed through parallel feature extraction pathways: a **position-aware bidirectional LSTM encoder** captures long-range sequential dependencies, while a **multi-scale convolutional module** detects local RNA motifs. These distinct feature sets are dynamically integrated via a learnable **multi-modal attention mechanism** before propagating through a hybrid graph neural network utilizing `TransformerConv` and `GATConv` layers to rigorously model non-local molecular interactions.

To ensure thermodynamic plausibility, the learned graph representation is hierarchically pooled and explicitly concatenated with projected thermodynamic features, creating a comprehensive molecular embedding that drives multi-task prediction heads for robust efficacy classification and uncertainty-quantified regression.

### Key Capabilities
* **Graph Neural Networks (GNNs):** Models the siRNA-mRNA duplex as a graph to capture base-pairing interactions.
* **Transformer Convolutions:** Utilizes `TransformerConv` layers to attend to long-range dependencies within the molecular graph structure.
* **Thermodynamic Integration:** Explicitly fuses 30 thermodynamic and structural features with deep learned representations to anchor predictions in physical reality.
* **Uncertainty Quantification:** Provides confidence scores alongside efficacy predictions using aleatoric uncertainty estimation.

## Dataset
The model relies on processed `.pkl` datasets containing siRNA-mRNA pairs.
- **Training:** Trained on data located at `Data/processed_data/train_data.pkl`.
- **Validation:** Evaluated during training using `Data/processed_data/val_data.pkl`.
- **Pretraining:** Supports self-supervised learning on larger unlabeled datasets (`pretrain_data.pkl`).
## Data Processing

### 1. Hybrid Feature Engineering
The preprocessing module (powered by `RiNALMo` and `ThermodynamicFeatureExtractor`) extracts:

* **Foundation Model Embeddings:** Uses `multimolecule/rinalmo-giga` to generate rich initial node embeddings.
* **Thermodynamic & Structural Features (30 dimensions):** Explicitly calculates 30 physicochemical features, including:
    * **Energy Profiles:** Terminal Dinucleotide Free Energy ($\Delta G$) and Enthalpy ($\Delta H$) for 5' and 3' ends.
    * **Stability Metrics:** Total Enthalpy ($\Delta H_{all}$), Melting Temperature ($T_m$), and End Differential Stability.
    * **Composition:** Overall GC content, Seed Region (positions 2-8) GC content, and Dinucleotide frequencies.
    * **Positional Motifs:** Binary indicators for efficacy-enhancing motifs (e.g., A at pos 6, U at pos 10, absence of G at pos 13) and specific nucleotide identities at terminals.
* **Positional Features:** One-hot encodings combined with position-aware embeddings.

### 2. Graph Construction
The siRNA-mRNA interaction is modeled as a graph:
* **Nodes:** Nucleotides in the duplex.
* **Edges:**
    * *Backbone Edges:* Connect adjacent nucleotides ($i \rightarrow i+1$).
    * *Interaction Edges:* Connect base pairs between siRNA and mRNA.
* **Edge Attributes (14 dimensions):** A comprehensive feature vector for every edge encoding:
    * **Connection Type:** Distinguishes between backbone phosphate bonds and hydrogen-bonded base pairs.
    * **Pairing Geometry:** One-hot encoding of pairing types (AU, GC, or Wobble GU).
    * **Interaction Strength:** Stability scores (GC=1.0, AU=0.8, Wobble=0.6, Mismatch=0.3).
    * **Canonical Status:** Binary flags for Watson-Crick vs. Wobble pairs.
    * **Positional Context:** Normalized position within the sequence and explicit Seed Region indicators (positions 2-8).

## Architecture Overview

![Architecture](Architecture.jpg)

### 1. Multi-Modal Encoder
- **Position-Aware Encoder:** Bi-directional LSTM with learnable position embeddings.
- **Motif Detector:** Multi-scale 1D Convolutional layers (Kernels: 3, 5, 7, 9) to detect sequence motifs.
- **MultiModal Attention:** A specialized attention mechanism to fuse Sequence, Motif, and Thermodynamic features adaptively.

### 2. Graph Processing Layers
The core `OGencoder` utilizes a hybrid GNN block repeated 8 times:
- **TransformerConv:** Captures global graph context with multi-head attention.
- **GATConv:** Graph Attention Network for local neighborhood aggregation.
- **Residual Connections & LayerNorm:** Ensures stable gradient flow.
- **Hierarchical Pooling:** Aggregates node features into a graph-level representation.

### 3. Prediction Heads
- **Regression Head:** Predicts efficacy score (0-1) and uncertainty variance.
- **Classification Head:** Binary prediction (Effective vs. Ineffective).


## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (optional, but recommended)

#### Step 1: Clone the Repository

```bash
git clone https://github.com/drugparadigm/OliGraphs.git
cd OliGraphs
```

#### Step 2: Set Up Python Environment
Choose either Conda (Option A) or Virtual Environment (Option B):
Option A: Conda Environment

```bash
conda create --name oligraphs python=3.9.0
conda activate oligraphs
```
Option B: Virtual Environment

```bash
python3.9 -m venv oligraphs
```

On Windows:
```
oligraphs\Scripts\activate
```
On macOS/Linux:
```
source oligraphs/bin/activate
```

#### Step 3: Install dependencies
```
pip install -r requirements.txt
```
Install PyTorch based on your hardware:
For CPU:

```bash
pip install torch 
```
For GPU (CUDA support):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126

```

## Usage 

### 1. Preprocessing 

Process raw CSV files into graph objects with thermodynamic features(file in Data folder):

```bash
python Preprocess.py
```

### 2. Self-Supervised Pretraining (Optional)

Pretrain the model using contrastive learning and masked node prediction:

```bash
python pretrain.py
```

### 3. Supervised Training
Train the model on labeled data with optional pretrained weights:

```bash
python train.py
```

### Training Parameters

| Parameter | Description | Value |
|-----------|-------------|-------|
| `batch_size` | Training batch size | `64` |
| `learning_rate` | Differential learning rates (Transformer/Heads) | `1e-4` to `2e-4` |
| `num_epochs` | Maximum training epochs | `150` |
| `hidden_dim` | Hidden feature dimension | `512` |
| `num_layers` | Number of GNN/TransformerConv layers | `8` |
| `num_heads` | Number of attention heads | `8` |
| `dropout` | Dropout rate | `0.15` |
| `foundation_dim` | Dimension of RiNALMo embeddings | `1280` |
| `warmup_epochs` | Learning rate warmup period | `10` |

#### Training Output

Best model: Checkpoints/training/best_enhanced_model.pt
Periodic checkpoints: Checkpoints/training/checkpoint_epoch_*.pt

```bash
Time       Epoch  Train Loss   Train PCC  Val Acc    Val AUC-ROC  Val AUC-PR   Val PCC    Top-10    LR        
00:12:34   50     0.3245       0.7234     0.8456     0.8912       0.8734       0.7543     0.7234    0.000100
```

### 4. Model Evaluation
Evaluate the trained model on test data:

```bash
python test.py
```

#### Testing Output
```
================================================================================
CURRENT TEST METRICS
================================================================================
  F1 Score:             0.8417
  AUC-ROC:              0.8257
  AUC-PR:               0.8260
  PCC (Pearson):        0.6150
  MSE (Regression):     0.8902
================================================================================
```

### 5. Inference

#### Option A: Single Sequence Prediction
Predict efficacy for a single siRNA-mRNA pair:

```bash
python inference.py \
  --mode string \
  --model-path Checkpoints/training/best_enhanced_model.pt \
  --sirna "UGAGGUAGUAGGUUGUAUAGUU" \
  --mrna "UAUACAACCUACUACCUCAUU" \
  --device cuda:2
```

**Example Output:**
```
================================================================================
Prediction Results
================================================================================
siRNA Sequence:        UGAGGUAGUAGGUUGUAUA
mRNA Sequence:         UAUACAACCUACUACCUCA
--------------------------------------------------------------------------------
Efficacy Score:        0.8532
Confidence Score:      0.9234
Uncertainty (Var):     0.0123
--------------------------------------------------------------------------------
Classification:        Effective (Class 1)
  Class 0 Probability: 0.1234
  Class 1 Probability: 0.8766
================================================================================
```

#### Option B: Batch FASTA File Prediction
Predict efficacy for multiple sequences from FASTA files:

```bash
python inference.py \
  --mode fasta \
  --model-path Checkpoints/training/best_enhanced_model.pt \
  --sirna-fasta sequences/sirna.fasta \
  --mrna-fasta sequences/mrna.fasta \
  --output results.csv \
  --device cuda:2
```

FASTA File Format:
```
>siRNA_1
UGAGGUAGUAGGUUGUAUAGUU
>siRNA_2
GACUACGAGUACGACUAGCUU
>siRNA_3
CAUCGAGCUAGCUAGCUAGUU
```

#### Inference Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--mode` | Input mode: `string` or `fasta` | Required |
| `--model-path` | Path to trained model | `Checkpoints/best_enhanced_model.pt` |
| `--device` | Device (`cuda:0`, `cuda:1`, `cpu`) | `cuda:2` |
| `--sirna` | siRNA sequence (string mode) | Required for string |
| `--mrna` | mRNA sequence (string mode) | Required for string |
| `--sirna-fasta` | siRNA FASTA file (fasta mode) | Required for fasta |
| `--mrna-fasta` | mRNA FASTA file (fasta mode) | Required for fasta |
| `--output` | Output CSV path (fasta mode) | Optional |
| `--no-checkpoint-info` | Disable checkpoint info display | `False` |



#### Output CSV Format:

```
sirna_id,mrna_id,sirna,mrna,efficacy,confidence_score,uncertainty_variance,classification,class_0_probability,class_1_probability
siRNA_1,mRNA_1,UGAGGUAGUAGGUUGUAUA,UAUACAACCUACUACCUCA,0.8532,0.9234,0.0123,1,0.1234,0.8766
siRNA_2,mRNA_2,GACUACGAGUACGACUAGC,GCUAGUCGUACUCGUAGUC,0.3421,0.8567,0.0234,0,0.6789,0.3211
```

### Sequence Requirements

- **Nucleotides**: Only A, U, G, C allowed (T automatically converted to U)
- **siRNA length**: Minimum 19 nucleotides (longer sequences truncated to 19)
- **mRNA length**: Minimum 19 nucleotides (automatically finds binding site)
- **Case**: Automatically converted to uppercase


## Troubleshooting

### Common Issues
**Issue**: RuntimeError: CUDA out of memory
Solution: Reduce batch size or switch to CPU
```bash
python train.py --batch_size 32
python inference.py --device cpu
```

**Issue**: ImportError: cannot import name 'OGencoder'
Solution: Ensure model.py is in the same directory or add to PYTHONPATH
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/Oligraphs"
```
**Issue**: FileNotFoundError: No such file or directory: 'Data/processed_data/train_data.pkl'
Solution: Run preprocessing first
```bash
python Preprocess.py
```
**Issue**: ModuleNotFoundError: No module named 'multimolecule'
Solution: Install multimolecule
```bash
pip install multimolecule==0.0.8
```
**Issue**: RiNALMo model download fails
Solution: Download manually and set cache directory
```bash
export HF_HOME=/path/to/cache
python -c "from multimolecule import RiNALMoModel; RiNALMoModel.from_pretrained('multimolecule/rinalmo-giga')"
```
**Issue**: Invalid checkpoint format
Solution: Verify checkpoint contains required keys
```
pythoncheckpoint = torch.load('checkpoint.pt')
print(checkpoint.keys())  # Should contain 'model_state_dict'
```

### Performance Optimization

**Multi-GPU training:**

Modify train.py
```bash
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
```


