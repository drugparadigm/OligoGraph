# import pandas as pd
# import numpy as np
# import torch
# from torch_geometric.data import Data
# import pickle
# import os
# import random
# from typing import List, Tuple, Dict
# import logging
# import warnings


# from multimolecule import RnaTokenizer, RiNALMoModel

# warnings.filterwarnings('ignore')
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


# class ThermodynamicFeatureExtractor:
#     """
#     Extracts 24 thermodynamic and positional features from siRNA sequences.
#     """

#     def __init__(self):
#         # RNA nearest-neighbor thermodynamic parameters (SantaLucia/Turner)
#         # Values in kcal/mol at 37°C
#         self.dinucleotide_dg = {
#             'AA': -0.9, 'AU': -0.9, 'UA': -0.9, 'UU': -0.9,
#             'GA': -1.3, 'GU': -1.3, 'UG': -1.3, 'AG': -1.3,
#             'CA': -1.7, 'CU': -1.7, 'UC': -1.7, 'AC': -1.7,
#             'GG': -2.9, 'GC': -2.1, 'CG': -1.5, 'CC': -2.9
#         }

#         self.dinucleotide_dh = {
#             'AA': -6.6, 'AU': -5.7, 'UA': -8.1, 'UU': -6.6,
#             'GA': -8.8, 'GU': -10.5, 'UG': -10.5, 'AG': -7.6,
#             'CA': -10.4, 'CU': -7.6, 'UC': -8.8, 'AC': -10.2,
#             'GG': -13.0, 'GC': -14.2, 'CG': -10.1, 'CC': -13.0
#         }

#         # These values will be used for z-score normalization
#         self.thermo_mean = np.array([-1.5, -1.5, -1.5, -1.5, -8.0, -50.0, 0.0])
#         self.thermo_std = np.array([0.8, 0.8, 0.8, 0.8, 3.0, 15.0, 2.0])

#     def get_dinucleotide_dg(self, dinuc: str) -> float:
#         """Get Gibbs free energy for a dinucleotide."""
#         return self.dinucleotide_dg.get(dinuc, -1.0)  

#     def get_dinucleotide_dh(self, dinuc: str) -> float:
#         """Get enthalpy for a dinucleotide."""
#         return self.dinucleotide_dh.get(dinuc, -8.0)  

#     def calculate_dh_all(self, sequence: str) -> float:
#         """Calculate total enthalpy of the entire sequence."""
#         total_dh = 0.0
#         for i in range(len(sequence) - 1):
#             dinuc = sequence[i:i+2]
#             total_dh += self.get_dinucleotide_dh(dinuc)
#         return total_dh

#     def extract_features(self, siRNA: str) -> torch.Tensor:
#         """
#         Extract all 24 thermodynamic and positional features from siRNA sequence.
#         """
#         seq = siRNA.upper()
#         seq_len = len(seq)
#         if seq_len == 0:
#             return torch.zeros(24, dtype=torch.float32)

#         features = []

#         # === Terminal Dinucleotide Gibbs Free Energy ===
#         dg_1 = self.get_dinucleotide_dg(seq[0:2]) if seq_len >= 2 else 0.0
#         features.append(dg_1)
#         dg_2 = self.get_dinucleotide_dg(seq[1:3]) if seq_len >= 3 else 0.0
#         features.append(dg_2)
#         dg_13 = self.get_dinucleotide_dg(seq[12:14]) if seq_len >= 14 else 0.0
#         features.append(dg_13)
#         dg_18 = self.get_dinucleotide_dg(seq[-2:]) if seq_len >= 2 else 0.0
#         features.append(dg_18)

#         # === Terminal Dinucleotide Enthalpy ===
#         dh_1 = self.get_dinucleotide_dh(seq[0:2]) if seq_len >= 2 else 0.0
#         features.append(dh_1)

#         # === Overall Thermodynamic Properties ===
#         dh_all = self.calculate_dh_all(seq)
#         features.append(dh_all)
#         dg_5prime = (dg_1 + dg_2) / 2.0
#         dg_3prime = (dg_13 + dg_18) / 2.0
#         ends = (dg_5prime - dg_3prime) * 1.5
#         features.append(ends)

#         # === Normalize thermodynamic features (z-score) ===
#         thermo_features = np.array(features)
#         thermo_features = (thermo_features - self.thermo_mean) / (self.thermo_std + 1e-6)
#         features = thermo_features.tolist()

#         # === Positional Nucleotide Identity (binary flags) ===
#         features.append(1.0 if seq[0] == 'U' else 0.0)
#         features.append(1.0 if seq[0] == 'G' else 0.0)
#         features.append(1.0 if seq[0] == 'C' else 0.0)
#         features.append(1.0 if seq_len >= 2 and seq[1] == 'U' else 0.0)
#         features.append(1.0 if seq_len >= 19 and seq[18] == 'A' else 0.0)

#         # === Positional Dinucleotide Identity (binary flags) ===
#         first_dinuc = seq[0:2] if seq_len >= 2 else ''
#         features.append(1.0 if first_dinuc == 'UU' else 0.0)
#         features.append(1.0 if first_dinuc == 'GG' else 0.0)
#         features.append(1.0 if first_dinuc == 'GC' else 0.0)
#         features.append(1.0 if first_dinuc == 'CC' else 0.0)
#         features.append(1.0 if first_dinuc == 'CG' else 0.0)

#         # === Global Sequence Composition (proportions) ===
#         features.append(seq.count('U') / seq_len)
#         features.append(seq.count('G') / seq_len)
#         dinuc_count = seq_len - 1 if seq_len > 1 else 1
#         features.append(sum(1 for i in range(seq_len - 1) if seq[i:i+2] == 'GG') / dinuc_count)
#         features.append(sum(1 for i in range(seq_len - 1) if seq[i:i+2] == 'UA') / dinuc_count)
#         features.append(sum(1 for i in range(seq_len - 1) if seq[i:i+2] == 'CC') / dinuc_count)
#         features.append(sum(1 for i in range(seq_len - 1) if seq[i:i+2] == 'GC') / dinuc_count)
#         features.append(sum(1 for i in range(seq_len - 1) if seq[i:i+2] == 'UU') / dinuc_count)

#         return torch.tensor(features, dtype=torch.float32)

#     def extract_extended_sirna_features(self, sirna: str) -> torch.Tensor:
#         """
#         NEW: Extracts additional structural and motif-based features for an siRNA.
#         """
#         seq = sirna.upper()
#         seq_len = len(seq)
#         if seq_len == 0:
#             return torch.zeros(6, dtype=torch.float32)

#         # 1. Melting Temperature (Tm) using a basic formula
#         tm = (seq.count('A') + seq.count('U')) * 2 + (seq.count('G') + seq.count('C')) * 4

#         # 2. Overall GC Content
#         gc_all = (seq.count('G') + seq.count('C')) / seq_len if seq_len > 0 else 0

#         # 3. Seed Region GC Content (positions 2-8, 1-indexed)
#         seed_seq = seq[1:8]
#         gc_seed = (seed_seq.count('G') + seed_seq.count('C')) / len(seed_seq) if len(seed_seq) > 0 else 0

#         # 4. Efficacy-associated motifs
#         motif_A6 = 1.0 if seq_len >= 6 and seq[5] == 'A' else 0.0
#         motif_U10 = 1.0 if seq_len >= 10 and seq[9] == 'U' else 0.0
#         motif_no_G13 = 1.0 if seq_len >= 13 and seq[12] != 'G' else 0.0

#         extended_features = [tm, gc_all, gc_seed, motif_A6, motif_U10, motif_no_G13]
#         return torch.tensor(extended_features, dtype=torch.float32)


# class SimplifiedsiRNADataProcessor:
#     def __init__(self):
#         self.nucleotide_to_idx = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
#         self.thermo_extractor = ThermodynamicFeatureExtractor()
#         logger.info("Initialized thermodynamic and extended feature extractor")

#         logger.info("Loading RiNALMo model and tokenizer...")
#         self.model_name = "multimolecule/rinalmo-giga"
#         self.tokenizer = RnaTokenizer.from_pretrained(self.model_name)
#         self.rinalmo_model = RiNALMoModel.from_pretrained(self.model_name)
#         self.rinalmo_model.eval()

#         self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
#         self.rinalmo_model = self.rinalmo_model.to(self.device)
#         logger.info(f"RiNALMo model loaded on {self.device}")

#     def get_rinalmo_embeddings(self, sequence: str) -> torch.Tensor:
#         try:
#             with torch.no_grad():
#                 tokenized = self.tokenizer(
#                     sequence, return_tensors="pt", padding=True,
#                     truncation=True, max_length=512
#                 ).to(self.device)
#                 outputs = self.rinalmo_model(**tokenized)
#                 embeddings = outputs.last_hidden_state.squeeze(0)

#                 if embeddings.shape[0] > 2:
#                     embeddings = embeddings[1:-1]

#                 if embeddings.shape[0] != len(sequence):
#                     if embeddings.shape[0] > len(sequence):
#                         embeddings = embeddings[:len(sequence)]
#                     else:
#                         padding_size = len(sequence) - embeddings.shape[0]
#                         padding = embeddings[-1:].repeat(padding_size, 1)
#                         embeddings = torch.cat([embeddings, padding], dim=0)

#                 return embeddings.cpu()
#         except Exception as e:
#             logger.warning(
#                 f"Failed to get RiNALMo embeddings for sequence of length {len(sequence)}: {e}. "
#                 "Returning random tensor."
#             )
#             return torch.randn(len(sequence), 1280)

#     def one_hot_encode(self, sequence: str) -> torch.Tensor:
#         """One-hot encode an RNA sequence."""
#         encoding = []
#         for nucleotide in sequence:
#             one_hot = [0, 0, 0, 0]
#             if nucleotide in self.nucleotide_to_idx:
#                 one_hot[self.nucleotide_to_idx[nucleotide]] = 1
#             encoding.append(one_hot)
#         return torch.tensor(encoding, dtype=torch.float)

#     def compute_base_pair_features(self, nuc1: str, nuc2: str, position: int,
#                                      guide_length: int) -> List[float]:
#         """Compute detailed base pair features for graph edges."""
#         features = [0.0] * 14
#         features[1] = 1.0  # Base pairing indicator
#         bp_types = {('A','U'): 2, ('U','A'): 2, ('G','C'): 3, ('C','G'): 3, ('G','U'): 4, ('U','G'): 4}
#         bp_type_idx = bp_types.get((nuc1, nuc2), 5)
#         if bp_type_idx < 6:
#             features[bp_type_idx] = 1.0
#         features[6] = position / guide_length if guide_length > 0 else 0
#         canonical_pairs = {('A','U'), ('U','A'), ('G','C'), ('C','G')}
#         features[7] = 1.0 if (nuc1, nuc2) in canonical_pairs else 0
#         wobble_pairs = {('G','U'), ('U','G')}
#         features[8] = 1.0 if (nuc1, nuc2) in wobble_pairs else 0
#         stability_scores = {('G','C'): 1.0, ('C','G'): 1.0, ('A','U'): 0.8, ('U','A'): 0.8, ('G','U'): 0.6, ('U','G'): 0.6}
#         features[9] = stability_scores.get((nuc1, nuc2), 0.3)
#         features[10] = 1.0 if 2 <= position <= 8 else 0
#         return features

#     def create_graph_structure(self, siRNA: str, mRNA: str) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Create the graph structure (edges and edge features)."""
#         guide_len, target_len = len(siRNA), len(mRNA)
#         total_len = guide_len + target_len
#         edges, edge_features = [], []

#         for i in range(total_len - 1):
#             edges.extend([[i, i + 1], [i + 1, i]])
#             backbone_feat = [1.0] + [0.0] * 13
#             edge_features.extend([backbone_feat, backbone_feat])

#         for i in range(min(guide_len, target_len)):
#             guide_pos, target_pos = i, guide_len + (target_len - 1 - i)
#             edges.extend([[guide_pos, target_pos], [target_pos, guide_pos]])
#             bp_feat = self.compute_base_pair_features(siRNA[i], mRNA[i], i, guide_len)
#             edge_features.extend([bp_feat, bp_feat])

#         if not edges:
#             return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 14), dtype=torch.float)

#         edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
#         edge_attr = torch.tensor(edge_features, dtype=torch.float)

#         valid_mask = (edge_index < total_len).all(dim=0)
#         if not valid_mask.all():
#             edge_index = edge_index[:, valid_mask]
#             edge_attr = edge_attr[valid_mask]

#         return edge_index, edge_attr

#     def create_data_object(self, siRNA: str, mRNA: str, label: float, y_class: int) -> Data:
#         """
#         Create a single PyTorch Geometric Data object with a total of 30 features.
#         """
#         duplex = siRNA + mRNA
#         num_nodes = len(duplex)

#         foundation_features = self.get_rinalmo_embeddings(duplex)
#         onehot_features = self.one_hot_encode(duplex)
#         positions = torch.arange(num_nodes)
#         edge_index, edge_attr = self.create_graph_structure(siRNA, mRNA)
#         thermodynamic_features = self.thermo_extractor.extract_features(siRNA)
#         extended_features = self.thermo_extractor.extract_extended_sirna_features(siRNA)
#         all_sirna_features = torch.cat([thermodynamic_features, extended_features])
#         thermo_node_features = torch.zeros(num_nodes, 30)
#         sirna_len = len(siRNA)
#         thermo_node_features[:sirna_len] = all_sirna_features.unsqueeze(0).expand(sirna_len, -1)
#         data = Data(
#             x=foundation_features.float(),
#             foundation_features=foundation_features.float(),
#             onehot_features=onehot_features.float(),
#             positions=positions.long(),
#             edge_index=edge_index.long(),
#             edge_attr=edge_attr.float(),
#             y=torch.tensor(label, dtype=torch.float),
#             y_class=torch.tensor(y_class, dtype=torch.long),
#             thermodynamic_features=thermo_node_features.float(),
#             num_nodes=num_nodes
#         )

#         if data.edge_index.numel() > 0 and data.edge_index.max() >= num_nodes:
#             logger.warning(f"Correcting invalid edge index for sequence of length {num_nodes}")
#             valid_mask = (data.edge_index < num_nodes).all(dim=0)
#             data.edge_index = data.edge_index[:, valid_mask]
#             data.edge_attr = data.edge_attr[valid_mask]

#         return data

#     def process_dataset(self, df: pd.DataFrame) -> List[Data]:
#         logger.info(f"Processing dataset with {len(df)} samples...")
#         data_objects = []

#         for idx, row in df.iterrows():
#             try:
#                 sirna_seq = str(row['siRNA']).upper().replace('T', 'U')
#                 mrna_seq = str(row['mRNA']).upper().replace('T', 'U')

#                 data_obj = self.create_data_object(
#                     sirna_seq, mrna_seq, float(row['label']), int(row['y'])
#                 )
#                 data_objects.append(data_obj)

#                 if (idx + 1) % 500 == 0:
#                     logger.info(f" ...processed {idx + 1}/{len(df)} samples.")
#             except Exception as e:
#                 logger.warning(f"Skipping row {idx} due to error: {e}")
#                 continue

#         logger.info(f"Successfully processed {len(data_objects)}/{len(df)} samples.")
#         return data_objects


# def main():
#     logger.info("=" * 50)
#     logger.info("Starting Preprocessing with 30 siRNA features and Graph Strucutre")
#     logger.info("=" * 50)

#     processor = SimplifiedsiRNADataProcessor()

#     try:
#         hu_df = pd.read_csv('Hu.csv')
#         mix_df = pd.read_csv('Mix.csv')
#         taka_df = pd.read_csv('Taka.csv')
#     except FileNotFoundError as e:
#         logger.error(f"FATAL: Could not find input file: {e}. Aborting.")
#         return

#     required_cols = {'siRNA', 'mRNA', 'label', 'y'}
#     if not required_cols.issubset(hu_df.columns) or not required_cols.issubset(mix_df.columns):
#         logger.error("FATAL: Input CSVs must contain 'siRNA', 'mRNA', 'label', and 'y' columns.")
#         return

#     logger.info("CSV column check passed.")

#     train_data = processor.process_dataset(hu_df)
#     val_data = processor.process_dataset(mix_df)
#     val_data_2 = processor.process_dataset(taka_df)

#     os.makedirs('processed_data', exist_ok=True)
#     with open('processed_data/train_data.pkl', 'wb') as f:
#         pickle.dump(train_data, f)
#     with open('processed_data/val_data.pkl', 'wb') as f:
#         pickle.dump(val_data, f)
#     with open('processed_data/val_data_2.pkl', 'wb') as f:
#         pickle.dump(val_data_2, f)

#     logger.info("=" * 50)
#     logger.info("Data processing completed successfully!")
#     logger.info(f"Saved training data: {len(train_data)} samples to Data/processed_data/train_data.pkl")
#     logger.info(f"Saved validation data: {len(val_data)} samples to Data/processed_data/val_data.pkl")
#     logger.info(f"Saved validation data 2: {len(val_data_2)} samples to Data/processed_data/val_data_2.pkl")
#     logger.info("Each sample now includes 30 thermodynamic and structural features!")
#     logger.info("=" * 50)


# if __name__ == "__main__":
#     main()

# Import necessary libraries
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import pickle
import os
import random
from typing import List, Tuple, Dict
import logging
import warnings

# RiNALMo imports for foundation model embeddings
from multimolecule import RnaTokenizer, RiNALMoModel

# --- Setup ---
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ThermodynamicFeatureExtractor:
    """
    Extracts 24 thermodynamic and positional features from siRNA sequences.
    """

    def __init__(self):
        # RNA nearest-neighbor thermodynamic parameters (SantaLucia/Turner)
        # Values in kcal/mol at 37°C
        self.dinucleotide_dg = {
            'AA': -0.9, 'AU': -0.9, 'UA': -0.9, 'UU': -0.9,
            'GA': -1.3, 'GU': -1.3, 'UG': -1.3, 'AG': -1.3,
            'CA': -1.7, 'CU': -1.7, 'UC': -1.7, 'AC': -1.7,
            'GG': -2.9, 'GC': -2.1, 'CG': -1.5, 'CC': -2.9
        }

        self.dinucleotide_dh = {
            'AA': -6.6, 'AU': -5.7, 'UA': -8.1, 'UU': -6.6,
            'GA': -8.8, 'GU': -10.5, 'UG': -10.5, 'AG': -7.6,
            'CA': -10.4, 'CU': -7.6, 'UC': -8.8, 'AC': -10.2,
            'GG': -13.0, 'GC': -14.2, 'CG': -10.1, 'CC': -13.0
        }

        # Normalization statistics (pre-computed from training data)
        # These values will be used for z-score normalization
        self.thermo_mean = np.array([-1.5, -1.5, -1.5, -1.5, -8.0, -50.0, 0.0])
        self.thermo_std = np.array([0.8, 0.8, 0.8, 0.8, 3.0, 15.0, 2.0])

    def get_dinucleotide_dg(self, dinuc: str) -> float:
        """Get Gibbs free energy for a dinucleotide."""
        return self.dinucleotide_dg.get(dinuc, -1.0)  # Default value for unknown

    def get_dinucleotide_dh(self, dinuc: str) -> float:
        """Get enthalpy for a dinucleotide."""
        return self.dinucleotide_dh.get(dinuc, -8.0)  # Default value for unknown

    def calculate_dh_all(self, sequence: str) -> float:
        """Calculate total enthalpy of the entire sequence."""
        total_dh = 0.0
        for i in range(len(sequence) - 1):
            dinuc = sequence[i:i+2]
            total_dh += self.get_dinucleotide_dh(dinuc)
        return total_dh

    def extract_features(self, siRNA: str) -> torch.Tensor:
        """
        Extract all 24 thermodynamic and positional features from siRNA sequence.
        """
        seq = siRNA.upper()
        seq_len = len(seq)
        if seq_len == 0:
            return torch.zeros(24, dtype=torch.float32)

        features = []

        # === Terminal Dinucleotide Gibbs Free Energy ===
        dg_1 = self.get_dinucleotide_dg(seq[0:2]) if seq_len >= 2 else 0.0
        features.append(dg_1)
        dg_2 = self.get_dinucleotide_dg(seq[1:3]) if seq_len >= 3 else 0.0
        features.append(dg_2)
        dg_13 = self.get_dinucleotide_dg(seq[12:14]) if seq_len >= 14 else 0.0
        features.append(dg_13)
        dg_18 = self.get_dinucleotide_dg(seq[-2:]) if seq_len >= 2 else 0.0
        features.append(dg_18)

        # === Terminal Dinucleotide Enthalpy ===
        dh_1 = self.get_dinucleotide_dh(seq[0:2]) if seq_len >= 2 else 0.0
        features.append(dh_1)

        # === Overall Thermodynamic Properties ===
        dh_all = self.calculate_dh_all(seq)
        features.append(dh_all)
        dg_5prime = (dg_1 + dg_2) / 2.0
        dg_3prime = (dg_13 + dg_18) / 2.0
        ends = (dg_5prime - dg_3prime) * 1.5
        features.append(ends)

        # === Normalize thermodynamic features (z-score) ===
        thermo_features = np.array(features)
        thermo_features = (thermo_features - self.thermo_mean) / (self.thermo_std + 1e-6)
        features = thermo_features.tolist()

        # === Positional Nucleotide Identity (binary flags) ===
        features.append(1.0 if seq[0] == 'U' else 0.0)
        features.append(1.0 if seq[0] == 'G' else 0.0)
        features.append(1.0 if seq[0] == 'C' else 0.0)
        features.append(1.0 if seq_len >= 2 and seq[1] == 'U' else 0.0)
        features.append(1.0 if seq_len >= 19 and seq[18] == 'A' else 0.0)

        # === Positional Dinucleotide Identity (binary flags) ===
        first_dinuc = seq[0:2] if seq_len >= 2 else ''
        features.append(1.0 if first_dinuc == 'UU' else 0.0)
        features.append(1.0 if first_dinuc == 'GG' else 0.0)
        features.append(1.0 if first_dinuc == 'GC' else 0.0)
        features.append(1.0 if first_dinuc == 'CC' else 0.0)
        features.append(1.0 if first_dinuc == 'CG' else 0.0)

        # === Global Sequence Composition (proportions) ===
        features.append(seq.count('U') / seq_len)
        features.append(seq.count('G') / seq_len)
        dinuc_count = seq_len - 1 if seq_len > 1 else 1
        features.append(sum(1 for i in range(seq_len - 1) if seq[i:i+2] == 'GG') / dinuc_count)
        features.append(sum(1 for i in range(seq_len - 1) if seq[i:i+2] == 'UA') / dinuc_count)
        features.append(sum(1 for i in range(seq_len - 1) if seq[i:i+2] == 'CC') / dinuc_count)
        features.append(sum(1 for i in range(seq_len - 1) if seq[i:i+2] == 'GC') / dinuc_count)
        features.append(sum(1 for i in range(seq_len - 1) if seq[i:i+2] == 'UU') / dinuc_count)

        return torch.tensor(features, dtype=torch.float32)

    def extract_extended_sirna_features(self, sirna: str) -> torch.Tensor:
        """
        NEW: Extracts additional structural and motif-based features for an siRNA.
        """
        seq = sirna.upper()
        seq_len = len(seq)
        if seq_len == 0:
            return torch.zeros(6, dtype=torch.float32)

        # 1. Melting Temperature (Tm) using a basic formula
        tm = (seq.count('A') + seq.count('U')) * 2 + (seq.count('G') + seq.count('C')) * 4

        # 2. Overall GC Content
        gc_all = (seq.count('G') + seq.count('C')) / seq_len if seq_len > 0 else 0

        # 3. Seed Region GC Content (positions 2-8, 1-indexed)
        seed_seq = seq[1:8]
        gc_seed = (seed_seq.count('G') + seed_seq.count('C')) / len(seed_seq) if len(seed_seq) > 0 else 0

        # 4. Efficacy-associated motifs
        motif_A6 = 1.0 if seq_len >= 6 and seq[5] == 'A' else 0.0
        motif_U10 = 1.0 if seq_len >= 10 and seq[9] == 'U' else 0.0
        motif_no_G13 = 1.0 if seq_len >= 13 and seq[12] != 'G' else 0.0

        extended_features = [tm, gc_all, gc_seed, motif_A6, motif_U10, motif_no_G13]
        return torch.tensor(extended_features, dtype=torch.float32)


class SimplifiedsiRNADataProcessor:
    """
    Enhanced data processor with thermodynamic and extended feature extraction.
    """
    def __init__(self):
        self.nucleotide_to_idx = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
        self.thermo_extractor = ThermodynamicFeatureExtractor()
        logger.info("Initialized thermodynamic and extended feature extractor")

        logger.info("Loading RiNALMo model and tokenizer...")
        self.model_name = "multimolecule/rinalmo-giga"
        self.tokenizer = RnaTokenizer.from_pretrained(self.model_name)
        self.rinalmo_model = RiNALMoModel.from_pretrained(self.model_name)
        self.rinalmo_model.eval()

        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        self.rinalmo_model = self.rinalmo_model.to(self.device)
        logger.info(f"RiNALMo model loaded on {self.device}")

    def get_rinalmo_embeddings(self, sequence: str) -> torch.Tensor:
        """Get RiNALMo embeddings for a sequence, handling potential mismatches."""
        try:
            with torch.no_grad():
                tokenized = self.tokenizer(
                    sequence, return_tensors="pt", padding=True,
                    truncation=True, max_length=512
                ).to(self.device)
                outputs = self.rinalmo_model(**tokenized)
                embeddings = outputs.last_hidden_state.squeeze(0)

                if embeddings.shape[0] > 2:
                    embeddings = embeddings[1:-1]

                if embeddings.shape[0] != len(sequence):
                    if embeddings.shape[0] > len(sequence):
                        embeddings = embeddings[:len(sequence)]
                    else:
                        padding_size = len(sequence) - embeddings.shape[0]
                        padding = embeddings[-1:].repeat(padding_size, 1)
                        embeddings = torch.cat([embeddings, padding], dim=0)

                return embeddings.cpu()
        except Exception as e:
            logger.warning(
                f"Failed to get RiNALMo embeddings for sequence of length {len(sequence)}: {e}. "
                "Returning random tensor."
            )
            return torch.randn(len(sequence), 1280)

    def one_hot_encode(self, sequence: str) -> torch.Tensor:
        """One-hot encode an RNA sequence."""
        encoding = []
        for nucleotide in sequence:
            one_hot = [0, 0, 0, 0]
            if nucleotide in self.nucleotide_to_idx:
                one_hot[self.nucleotide_to_idx[nucleotide]] = 1
            encoding.append(one_hot)
        return torch.tensor(encoding, dtype=torch.float)

    def compute_base_pair_features(self, nuc1: str, nuc2: str, position: int,
                                     guide_length: int) -> List[float]:
        """Compute detailed base pair features for graph edges."""
        features = [0.0] * 14
        features[1] = 1.0  # Base pairing indicator
        bp_types = {('A','U'): 2, ('U','A'): 2, ('G','C'): 3, ('C','G'): 3, ('G','U'): 4, ('U','G'): 4}
        bp_type_idx = bp_types.get((nuc1, nuc2), 5)
        if bp_type_idx < 6:
            features[bp_type_idx] = 1.0
        features[6] = position / guide_length if guide_length > 0 else 0
        canonical_pairs = {('A','U'), ('U','A'), ('G','C'), ('C','G')}
        features[7] = 1.0 if (nuc1, nuc2) in canonical_pairs else 0
        wobble_pairs = {('G','U'), ('U','G')}
        features[8] = 1.0 if (nuc1, nuc2) in wobble_pairs else 0
        stability_scores = {('G','C'): 1.0, ('C','G'): 1.0, ('A','U'): 0.8, ('U','A'): 0.8, ('G','U'): 0.6, ('U','G'): 0.6}
        features[9] = stability_scores.get((nuc1, nuc2), 0.3)
        features[10] = 1.0 if 2 <= position <= 8 else 0
        return features

    def create_graph_structure(self, siRNA: str, mRNA: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create the graph structure (edges and edge features)."""
        guide_len, target_len = len(siRNA), len(mRNA)
        total_len = guide_len + target_len
        edges, edge_features = [], []

        for i in range(total_len - 1):
            edges.extend([[i, i + 1], [i + 1, i]])
            backbone_feat = [1.0] + [0.0] * 13
            edge_features.extend([backbone_feat, backbone_feat])

        for i in range(min(guide_len, target_len)):
            guide_pos, target_pos = i, guide_len + i
            # guide_pos, target_pos = i, guide_len + (target_len - 1 - i)
            edges.extend([[guide_pos, target_pos], [target_pos, guide_pos]])
            bp_feat = self.compute_base_pair_features(siRNA[i], mRNA[i], i, guide_len)
            edge_features.extend([bp_feat, bp_feat])

        if not edges:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 14), dtype=torch.float)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        valid_mask = (edge_index < total_len).all(dim=0)
        if not valid_mask.all():
            edge_index = edge_index[:, valid_mask]
            edge_attr = edge_attr[valid_mask]

        return edge_index, edge_attr

    def create_data_object(self, siRNA: str, mRNA: str, label: float, y_class: int) -> Data:
        """
        Create a single PyTorch Geometric Data object with a total of 30 features.
        """
        duplex = siRNA + mRNA
        num_nodes = len(duplex)

        foundation_features = self.get_rinalmo_embeddings(duplex)
        onehot_features = self.one_hot_encode(duplex)
        positions = torch.arange(num_nodes)
        edge_index, edge_attr = self.create_graph_structure(siRNA, mRNA)

        # === UPDATED: Extract and Combine All Features ===
        # Extract 24 original features
        thermodynamic_features = self.thermo_extractor.extract_features(siRNA)
        # Extract 6 new extended features
        extended_features = self.thermo_extractor.extract_extended_sirna_features(siRNA)
        # Combine into a single tensor of 30 features
        all_sirna_features = torch.cat([thermodynamic_features, extended_features])

        # Expand combined features to per-node format (total 30 features)
        thermo_node_features = torch.zeros(num_nodes, 30)
        sirna_len = len(siRNA)
        thermo_node_features[:sirna_len] = all_sirna_features.unsqueeze(0).expand(sirna_len, -1)
        # === END UPDATED SECTION ===

        data = Data(
            x=foundation_features.float(),
            foundation_features=foundation_features.float(),
            onehot_features=onehot_features.float(),
            positions=positions.long(),
            edge_index=edge_index.long(),
            edge_attr=edge_attr.float(),
            y=torch.tensor(label, dtype=torch.float),
            y_class=torch.tensor(y_class, dtype=torch.long),
            thermodynamic_features=thermo_node_features.float(),
            num_nodes=num_nodes
        )

        if data.edge_index.numel() > 0 and data.edge_index.max() >= num_nodes:
            logger.warning(f"Correcting invalid edge index for sequence of length {num_nodes}")
            valid_mask = (data.edge_index < num_nodes).all(dim=0)
            data.edge_index = data.edge_index[:, valid_mask]
            data.edge_attr = data.edge_attr[valid_mask]

        return data

    def process_dataset(self, df: pd.DataFrame) -> List[Data]:
        """Process an entire dataframe into a list of Data objects."""
        logger.info(f"Processing dataset with {len(df)} samples...")
        data_objects = []

        for idx, row in df.iterrows():
            try:
                sirna_seq = str(row['siRNA']).upper().replace('T', 'U')
                mrna_seq = str(row['mRNA']).upper().replace('T', 'U')

                data_obj = self.create_data_object(
                    sirna_seq, mrna_seq, float(row['label']), int(row['y'])
                )
                data_objects.append(data_obj)

                if (idx + 1) % 500 == 0:
                    logger.info(f" ...processed {idx + 1}/{len(df)} samples.")
            except Exception as e:
                logger.warning(f"Skipping row {idx} due to error: {e}")
                continue

        logger.info(f"Successfully processed {len(data_objects)}/{len(df)} samples.")
        return data_objects


def main():
    """Main function to load raw data, process it, and save the final pkl files."""
    logger.info("=" * 50)
    logger.info("Enhanced siRNA Data Processing with Thermodynamic Features")
    logger.info("=" * 50)

    processor = SimplifiedsiRNADataProcessor()

    try:
        hu_df = pd.read_csv('Hu.csv')
        # hu_df = pd.read_csv('Simone.csv')
        mix_df = pd.read_csv('Mix.csv')
        taka_df = pd.read_csv('Taka.csv')
    except FileNotFoundError as e:
        logger.error(f"FATAL: Could not find input file: {e}. Aborting.")
        return

    required_cols = {'siRNA', 'mRNA', 'label', 'y'}
    if not required_cols.issubset(hu_df.columns) or not required_cols.issubset(mix_df.columns):
        logger.error("FATAL: Input CSVs must contain 'siRNA', 'mRNA', 'label', and 'y' columns.")
        return

    logger.info("CSV column check passed.")

    train_data = processor.process_dataset(hu_df)
    val_data = processor.process_dataset(mix_df)
    val_data_2 = processor.process_dataset(taka_df)

    os.makedirs('processed_data', exist_ok=True)
    with open('processed_data/train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open('processed_data/val_data.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    with open('processed_data/val_data_2.pkl','wb') as f:
        pickle.dump(val_data_2,f)

    logger.info("=" * 50)
    logger.info("Data processing completed successfully!")
    logger.info(f"Saved training data: {len(train_data)} samples to processed_data/train_data.pkl")
    logger.info(f"Saved validation data: {len(val_data)} samples to processed_data/val_data.pkl")
    logger.info(f"Saved validation data: {len(val_data_2)} samples to processed_data/val_data_2.pkl")
    logger.info("Each sample now includes 30 thermodynamic and structural features!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()


