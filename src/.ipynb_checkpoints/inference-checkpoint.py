import torch
import numpy as np
import pandas as pd
import logging
import os
import argparse
from typing import Dict, Tuple, Optional, Any
from Bio import SeqIO

# Import from existing files
try:
    from model import OGencoder
except ImportError:
    print("FATAL: Could not import model.py. Ensure it is in the same directory.")
    exit(1)

# Import preprocessing dependencies
from multimolecule import RnaTokenizer, RiNALMoModel
from torch_geometric.data import Data

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ThermodynamicFeatureExtractor:
    """
    Extracts 24 thermodynamic and positional features from siRNA sequences.
    Replicated from Preprocess.py to avoid cross-folder imports.
    """

    def __init__(self):
        # RNA nearest-neighbor thermodynamic parameters
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

        self.thermo_mean = np.array([-1.5, -1.5, -1.5, -1.5, -8.0, -50.0, 0.0])
        self.thermo_std = np.array([0.8, 0.8, 0.8, 0.8, 3.0, 15.0, 2.0])

    def get_dinucleotide_dg(self, dinuc: str) -> float:
        return self.dinucleotide_dg.get(dinuc, -1.0)

    def get_dinucleotide_dh(self, dinuc: str) -> float:
        return self.dinucleotide_dh.get(dinuc, -8.0)

    def calculate_dh_all(self, sequence: str) -> float:
        total_dh = 0.0
        for i in range(len(sequence) - 1):
            dinuc = sequence[i:i+2]
            total_dh += self.get_dinucleotide_dh(dinuc)
        return total_dh

    def extract_features(self, siRNA: str) -> torch.Tensor:
        seq = siRNA.upper()
        seq_len = len(seq)
        if seq_len == 0:
            return torch.zeros(24, dtype=torch.float32)

        features = []

        # Terminal Dinucleotide Gibbs Free Energy
        dg_1 = self.get_dinucleotide_dg(seq[0:2]) if seq_len >= 2 else 0.0
        features.append(dg_1)
        dg_2 = self.get_dinucleotide_dg(seq[1:3]) if seq_len >= 3 else 0.0
        features.append(dg_2)
        dg_13 = self.get_dinucleotide_dg(seq[12:14]) if seq_len >= 14 else 0.0
        features.append(dg_13)
        dg_18 = self.get_dinucleotide_dg(seq[-2:]) if seq_len >= 2 else 0.0
        features.append(dg_18)

        # Terminal Dinucleotide Enthalpy
        dh_1 = self.get_dinucleotide_dh(seq[0:2]) if seq_len >= 2 else 0.0
        features.append(dh_1)

        # Overall Thermodynamic Properties
        dh_all = self.calculate_dh_all(seq)
        features.append(dh_all)
        dg_5prime = (dg_1 + dg_2) / 2.0
        dg_3prime = (dg_13 + dg_18) / 2.0
        ends = (dg_5prime - dg_3prime) * 1.5
        features.append(ends)

        # Normalize thermodynamic features
        thermo_features = np.array(features)
        thermo_features = (thermo_features - self.thermo_mean) / (self.thermo_std + 1e-6)
        features = thermo_features.tolist()

        # Positional Nucleotide Identity
        features.append(1.0 if seq[0] == 'U' else 0.0)
        features.append(1.0 if seq[0] == 'G' else 0.0)
        features.append(1.0 if seq[0] == 'C' else 0.0)
        features.append(1.0 if seq_len >= 2 and seq[1] == 'U' else 0.0)
        features.append(1.0 if seq_len >= 19 and seq[18] == 'A' else 0.0)

        # Positional Dinucleotide Identity
        first_dinuc = seq[0:2] if seq_len >= 2 else ''
        features.append(1.0 if first_dinuc == 'UU' else 0.0)
        features.append(1.0 if first_dinuc == 'GG' else 0.0)
        features.append(1.0 if first_dinuc == 'GC' else 0.0)
        features.append(1.0 if first_dinuc == 'CC' else 0.0)
        features.append(1.0 if first_dinuc == 'CG' else 0.0)

        # Global Sequence Composition
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
        seq = sirna.upper()
        seq_len = len(seq)
        if seq_len == 0:
            return torch.zeros(6, dtype=torch.float32)

        # Melting Temperature
        tm = (seq.count('A') + seq.count('U')) * 2 + (seq.count('G') + seq.count('C')) * 4

        # Overall GC Content
        gc_all = (seq.count('G') + seq.count('C')) / seq_len if seq_len > 0 else 0

        # Seed Region GC Content
        seed_seq = seq[1:8]
        gc_seed = (seed_seq.count('G') + seed_seq.count('C')) / len(seed_seq) if len(seed_seq) > 0 else 0

        # Efficacy-associated motifs
        motif_A6 = 1.0 if seq_len >= 6 and seq[5] == 'A' else 0.0
        motif_U10 = 1.0 if seq_len >= 10 and seq[9] == 'U' else 0.0
        motif_no_G13 = 1.0 if seq_len >= 13 and seq[12] != 'G' else 0.0

        extended_features = [tm, gc_all, gc_seed, motif_A6, motif_U10, motif_no_G13]
        return torch.tensor(extended_features, dtype=torch.float32)


class SimplifiedsiRNADataProcessor:
    """
    Simplified data processor for inference only.
    Replicated from Preprocess.py to avoid cross-folder imports.
    """
    
    def __init__(self, device='cuda:2'):
        self.nucleotide_to_idx = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
        self.thermo_extractor = ThermodynamicFeatureExtractor()
        logger.info("Initialized thermodynamic and extended feature extractor")

        logger.info("Loading RiNALMo model and tokenizer...")
        self.model_name = "multimolecule/rinalmo-giga"
        self.tokenizer = RnaTokenizer.from_pretrained(self.model_name)
        self.rinalmo_model = RiNALMoModel.from_pretrained(self.model_name)
        self.rinalmo_model.eval()

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.rinalmo_model = self.rinalmo_model.to(self.device)
        logger.info(f"RiNALMo model loaded on {self.device}")

    def get_rinalmo_embeddings(self, sequence: str) -> torch.Tensor:
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
        encoding = []
        for nucleotide in sequence:
            one_hot = [0, 0, 0, 0]
            if nucleotide in self.nucleotide_to_idx:
                one_hot[self.nucleotide_to_idx[nucleotide]] = 1
            encoding.append(one_hot)
        return torch.tensor(encoding, dtype=torch.float)

    def compute_base_pair_features(self, nuc1: str, nuc2: str, position: int,
                                     guide_length: int) -> list:
        features = [0.0] * 14
        features[1] = 1.0
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
        guide_len, target_len = len(siRNA), len(mRNA)
        total_len = guide_len + target_len
        edges, edge_features = [], []

        for i in range(total_len - 1):
            edges.extend([[i, i + 1], [i + 1, i]])
            backbone_feat = [1.0] + [0.0] * 13
            edge_features.extend([backbone_feat, backbone_feat])

        for i in range(min(guide_len, target_len)):
            guide_pos, target_pos = i, guide_len + (target_len - 1 - i)
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
        duplex = siRNA + mRNA
        num_nodes = len(duplex)

        foundation_features = self.get_rinalmo_embeddings(duplex)
        onehot_features = self.one_hot_encode(duplex)
        positions = torch.arange(num_nodes)
        edge_index, edge_attr = self.create_graph_structure(siRNA, mRNA)
        thermodynamic_features = self.thermo_extractor.extract_features(siRNA)
        extended_features = self.thermo_extractor.extract_extended_sirna_features(siRNA)
        all_sirna_features = torch.cat([thermodynamic_features, extended_features])
        thermo_node_features = torch.zeros(num_nodes, 30)
        sirna_len = len(siRNA)
        thermo_node_features[:sirna_len] = all_sirna_features.unsqueeze(0).expand(sirna_len, -1)
        
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


def display_checkpoint_info(checkpoint: Dict[str, Any]):
    """
    Extracts and displays metadata from the model checkpoint.
    """
    logger.info("\n" + "=" * 80)
    logger.info("CHECKPOINT INFORMATION")
    logger.info("=" * 80)
    
    # Display basic checkpoint info
    if 'epoch' in checkpoint:
        logger.info(f"  Checkpoint Epoch:     {checkpoint['epoch']}")
    
    if 'timestamp' in checkpoint:
        logger.info(f"  Saved at:             {checkpoint['timestamp']}")
    
    # Display training metrics if available
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        logger.info("\n--- Training Metrics (from checkpoint) ---")
        
        if 'train' in metrics:
            train_metrics = metrics['train']
            logger.info("  Training Set:")
            for key, value in train_metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"    {key:<20}: {value:.4f}")
                else:
                    logger.info(f"    {key:<20}: {value}")
        
        if 'val' in metrics:
            val_metrics = metrics['val']
            logger.info("\n  Validation Set:")
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"    {key:<20}: {value:.4f}")
                else:
                    logger.info(f"    {key:<20}: {value}")
    else:
        logger.warning("  No metrics found in checkpoint.")
    
    # Display optimizer info if available
    if 'optimizer_state_dict' in checkpoint:
        logger.info("\n  Optimizer state:      Present")
    
    # Display other available keys
    other_keys = [k for k in checkpoint.keys() if k not in ['epoch', 'timestamp', 'metrics', 'model_state_dict', 'optimizer_state_dict']]
    if other_keys:
        logger.info(f"\n  Other checkpoint keys: {', '.join(other_keys)}")
    
    logger.info("=" * 80)


class siRNAInference:
    """
    Inference engine for siRNA efficacy prediction.
    Handles both FASTA files and single string inputs.
    """
    
    def __init__(self, model_path: str, device: str = 'cuda:2', show_checkpoint_info: bool = True):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on
            show_checkpoint_info: Whether to display checkpoint information
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize preprocessor
        logger.info("Initializing preprocessor...")
        self.preprocessor = SimplifiedsiRNADataProcessor(device=device)
        
        # Load model
        logger.info(f"Loading model from {model_path}...")
        self.model, self.checkpoint = self._load_model(model_path, show_checkpoint_info)
        logger.info("Model loaded successfully!")
        
    def _load_model(self, model_path: str, show_info: bool = True) -> Tuple[torch.nn.Module, Dict]:
        """Load the trained model from checkpoint."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at '{model_path}'")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Display checkpoint information
        if show_info:
            display_checkpoint_info(checkpoint)
        
        # Initialize model with same architecture
        model = OGencoder(
            foundation_dim=1280,
            hidden_dim=512,
            num_heads=8,
            num_layers=8,
            dropout=0.15
        ).to(self.device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, checkpoint
    
    def find_complementary_site(self, sirna: str, mrna: str) -> Tuple[str, int]:
        """
        Find the complementary binding site of siRNA in mRNA.
        Uses reverse complement matching.
        
        Args:
            sirna: siRNA sequence (19nt)
            mrna: Full mRNA sequence
            
        Returns:
            Tuple of (matching_site, start_position)
        """
        # Create complement mapping
        complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
        
        # Get reverse complement of siRNA (this is what should match in mRNA)
        sirna_rc = ''.join([complement.get(base, 'N') for base in sirna[::-1]])
        
        # Search for best match in mRNA
        best_score = 0
        best_pos = 0
        window_size = len(sirna)
        
        for i in range(len(mrna) - window_size + 1):
            mrna_window = mrna[i:i+window_size]
            # Count matches
            matches = sum(1 for a, b in zip(sirna_rc, mrna_window) if a == b)
            if matches > best_score:
                best_score = matches
                best_pos = i
        
        # Extract the matching site
        matching_site = mrna[best_pos:best_pos+window_size]
        
        logger.info(f"Found binding site at position {best_pos} with {best_score}/{window_size} matches")
        
        return matching_site, best_pos
    
    def prepare_sequences(self, sirna: str, mrna: str) -> Tuple[str, str]:
        """
        Prepare siRNA and mRNA sequences for inference.
        - Truncates siRNA to first 19 nucleotides if longer
        - Finds complementary site in mRNA if longer than 19nt
        
        Args:
            sirna: siRNA sequence
            mrna: mRNA sequence
            
        Returns:
            Tuple of (processed_sirna, processed_mrna)
        """
        # Clean sequences
        sirna = sirna.upper().replace('T', 'U').strip()
        mrna = mrna.upper().replace('T', 'U').strip()
        
        # Process siRNA
        if len(sirna) > 19:
            logger.info(f"siRNA length ({len(sirna)}) > 19, taking first 19 nucleotides")
            sirna = sirna[:19]
        elif len(sirna) < 19:
            logger.warning(f"siRNA length ({len(sirna)}) < 19, this may affect prediction accuracy")
        
        # Process mRNA
        if len(mrna) > 19:
            logger.info(f"mRNA length ({len(mrna)}) > 19, finding complementary site")
            mrna, pos = self.find_complementary_site(sirna, mrna)
        elif len(mrna) < 19:
            logger.warning(f"mRNA length ({len(mrna)}) < 19, this may affect prediction accuracy")
        
        return sirna, mrna
    
    def predict_single(self, sirna: str, mrna: str) -> Dict:
        """
        Predict efficacy for a single siRNA-mRNA pair.
        
        Args:
            sirna: siRNA sequence
            mrna: mRNA sequence
            
        Returns:
            Dictionary containing prediction results
        """
        # Prepare sequences
        sirna_proc, mrna_proc = self.prepare_sequences(sirna, mrna)
        
        logger.info(f"Processing pair: siRNA={sirna_proc}, mRNA={mrna_proc}")
        
        # Create data object using preprocessor
        # Use dummy label and class for inference
        data = self.preprocessor.create_data_object(
            siRNA=sirna_proc,
            mRNA=mrna_proc,
            label=0.0,  # dummy
            y_class=0   # dummy
        )
        
        # Move to device
        data = data.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(data, mode="supervised")
        
        # Extract predictions
        classification_logits = outputs['classification'].cpu().numpy()[0]
        classification_probs = torch.softmax(outputs['classification'], dim=1).cpu().numpy()[0]
        classification_pred = int(np.argmax(classification_probs))
        
        regression_mean = outputs['regression_mean'].cpu().numpy()[0][0]
        regression_variance = outputs['regression_variance'].cpu().numpy()[0][0]
        
        # Calculate confidence score (inverse of uncertainty)
        # Lower variance = higher confidence
        confidence_score = 1.0 / (1.0 + regression_variance)
        
        results = {
            'sirna': sirna_proc,
            'mrna': mrna_proc,
            'efficacy': float(regression_mean),
            'confidence_score': float(confidence_score),
            'uncertainty_variance': float(regression_variance),
            'classification': classification_pred,
            'class_0_probability': float(classification_probs[0]),
            'class_1_probability': float(classification_probs[1])
        }
        
        return results
    
    def predict_from_strings(self, sirna: str, mrna: str) -> Dict:
        """
        Predict from single string inputs.
        
        Args:
            sirna: siRNA sequence string
            mrna: mRNA sequence string
            
        Returns:
            Dictionary containing prediction results
        """
        logger.info("=" * 80)
        logger.info("Running inference on single string input")
        logger.info("=" * 80)
        
        results = self.predict_single(sirna, mrna)
        
        self._print_results(results)
        
        return results
    
    def predict_from_fasta(self, sirna_fasta: str, mrna_fasta: str, output_csv: Optional[str] = None) -> pd.DataFrame:
        """
        Predict from FASTA files.
        
        Args:
            sirna_fasta: Path to siRNA FASTA file
            mrna_fasta: Path to mRNA FASTA file
            output_csv: Optional path to save results as CSV
            
        Returns:
            DataFrame containing all predictions
        """
        logger.info("=" * 80)
        logger.info("Running inference on FASTA files")
        logger.info("=" * 80)
        
        # Read FASTA files
        logger.info(f"Reading siRNA sequences from {sirna_fasta}")
        sirna_records = list(SeqIO.parse(sirna_fasta, "fasta"))
        
        logger.info(f"Reading mRNA sequences from {mrna_fasta}")
        mrna_records = list(SeqIO.parse(mrna_fasta, "fasta"))
        
        if len(sirna_records) != len(mrna_records):
            logger.warning(
                f"Number of siRNA sequences ({len(sirna_records)}) != "
                f"number of mRNA sequences ({len(mrna_records)})"
            )
        
        # Process all pairs
        all_results = []
        num_pairs = min(len(sirna_records), len(mrna_records))
        
        logger.info(f"Processing {num_pairs} siRNA-mRNA pairs...")
        
        for i in range(num_pairs):
            sirna_seq = str(sirna_records[i].seq)
            mrna_seq = str(mrna_records[i].seq)
            sirna_id = sirna_records[i].id
            mrna_id = mrna_records[i].id
            
            logger.info(f"\nProcessing pair {i+1}/{num_pairs}: {sirna_id} - {mrna_id}")
            
            try:
                results = self.predict_single(sirna_seq, mrna_seq)
                results['sirna_id'] = sirna_id
                results['mrna_id'] = mrna_id
                all_results.append(results)
                
                # Print individual results
                self._print_results(results, pair_num=i+1)
                
            except Exception as e:
                logger.error(f"Error processing pair {i+1}: {e}")
                continue
        
        # Create DataFrame
        df_results = pd.DataFrame(all_results)
        
        # Reorder columns
        column_order = [
            'sirna_id', 'mrna_id', 'sirna', 'mrna', 
            'efficacy', 'confidence_score', 'uncertainty_variance',
            'classification', 'class_0_probability', 'class_1_probability'
        ]
        df_results = df_results[column_order]
        
        # Save to CSV if requested
        if output_csv:
            df_results.to_csv(output_csv, index=False)
            logger.info(f"\nResults saved to {output_csv}")
        
        return df_results
    
    def _print_results(self, results: Dict, pair_num: Optional[int] = None):
        """Pretty print prediction results."""
        header = f"Results for Pair {pair_num}" if pair_num else "Prediction Results"
        logger.info("\n" + "=" * 80)
        logger.info(header)
        logger.info("=" * 80)
        logger.info(f"siRNA Sequence:        {results['sirna']}")
        logger.info(f"mRNA Sequence:         {results['mrna']}")
        logger.info("-" * 80)
        logger.info(f"Efficacy Score:        {results['efficacy']:.4f}")
        logger.info(f"Confidence Score:      {results['confidence_score']:.4f}")
        logger.info(f"Uncertainty (Var):     {results['uncertainty_variance']:.4f}")
        logger.info("-" * 80)
        logger.info(f"Classification:        {'Effective' if results['classification'] == 1 else 'Ineffective'} (Class {results['classification']})")
        logger.info(f"  Class 0 Probability: {results['class_0_probability']:.4f}")
        logger.info(f"  Class 1 Probability: {results['class_1_probability']:.4f}")
        logger.info("=" * 80)


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='siRNA Efficacy Prediction - Inference Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single string input
  python inference.py --mode string --sirna "GUGCCAUUGGAUGUUUAGA" --mrna "UCUAAACAUCCAAUGGCAC"
  
  # FASTA file input
  python inference.py --mode fasta --sirna-fasta sirna.fasta --mrna-fasta mrna.fasta --output results.csv
  
  # Disable checkpoint info display
  python inference.py --mode string --sirna "GUGCCAUUGGAUGUUUAGA" --mrna "UCUAAACAUCCAAUGGCAC" --no-checkpoint-info
        """
    )
    
    parser.add_argument('--mode', type=str, required=True, choices=['string', 'fasta'],
                        help='Input mode: "string" for single sequences or "fasta" for FASTA files')
    parser.add_argument('--model-path', type=str, default='Checkpoints/best_model.pt',
                        help='Path to model checkpoint (default: Checkpoints/best_model.pt)')
    parser.add_argument('--device', type=str, default='cuda:2',
                        help='Device to use for inference (default: cuda:2)')
    parser.add_argument('--no-checkpoint-info', action='store_true',
                        help='Disable checkpoint information display')
    
    # String mode arguments
    parser.add_argument('--sirna', type=str, help='siRNA sequence (required for string mode)')
    parser.add_argument('--mrna', type=str, help='mRNA sequence (required for string mode)')
    
    # FASTA mode arguments
    parser.add_argument('--sirna-fasta', type=str, help='Path to siRNA FASTA file (required for fasta mode)')
    parser.add_argument('--mrna-fasta', type=str, help='Path to mRNA FASTA file (required for fasta mode)')
    parser.add_argument('--output', type=str, help='Output CSV file path (optional for fasta mode)')
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == 'string':
        if not args.sirna or not args.mrna:
            parser.error("--sirna and --mrna are required for string mode")
    elif args.mode == 'fasta':
        if not args.sirna_fasta or not args.mrna_fasta:
            parser.error("--sirna-fasta and --mrna-fasta are required for fasta mode")
    
    # Initialize inference engine
    try:
        inference_engine = siRNAInference(
            model_path=args.model_path,
            device=args.device,
            show_checkpoint_info=not args.no_checkpoint_info
        )
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        return
    
    # Run inference based on mode
    try:
        if args.mode == 'string':
            results = inference_engine.predict_from_strings(args.sirna, args.mrna)
            
        elif args.mode == 'fasta':
            results = inference_engine.predict_from_fasta(
                args.sirna_fasta,
                args.mrna_fasta,
                args.output
            )
            logger.info("\n" + "=" * 80)
            logger.info("Summary Statistics")
            logger.info("=" * 80)
            logger.info(f"Total pairs processed:  {len(results)}")
            logger.info(f"Mean efficacy:          {results['efficacy'].mean():.4f}")
            logger.info(f"Mean confidence:        {results['confidence_score'].mean():.4f}")
            logger.info(f"Effective (Class 1):    {(results['classification'] == 1).sum()}")
            logger.info(f"Ineffective (Class 0):  {(results['classification'] == 0).sum()}")
            logger.info("=" * 80)
    
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    logger.info("\nInference completed successfully!")


if __name__ == "__main__":
    main()