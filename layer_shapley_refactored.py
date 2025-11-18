import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from architectures_unstructured import ARCHITECTURES, get_architecture
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer, Adam, AdamW
from torch.optim.lr_scheduler import StepLR
import datetime
import time
import numpy as np
import copy
import types
from tqdm.rich import tqdm
import time
from math import ceil, comb
from train_utils import AverageMeter, accuracy, accuracy_list, init_logfile, log
from scipy.special import comb, perm
from utils import *
import sys
import itertools
from archs_unstructured.cifar_resnet import BasicBlock, BasicBlock_IN, LearnableAlpha
from rich.console import Console
from random import sample
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

console = Console()


class ModelType(Enum):
    """Model architecture types"""
    STANDARD = "standard"
    WRN = "wrn"


@dataclass
class ShapleyConfig:
    """Configuration for Shapley value computation"""
    threshold: float = 1e-2
    early_termination_threshold: Optional[float] = 5.0
    max_subset_size: int = 4
    monte_carlo_samples: int = 100
    model_type: ModelType = ModelType.STANDARD
    
    def __post_init__(self):
        if isinstance(self.model_type, str):
            self.model_type = ModelType(self.model_type)


class LayerShapley:
    """
    Refactored Layer Shapley implementation for computing layer importance scores.
    
    This class provides a clean, reusable API for calculating Shapley values
    of neural network layers based on their contribution to model performance.
    """
    
    def __init__(self, model: nn.Module, device: torch.device, config: Optional[ShapleyConfig] = None):
        """
        Initialize LayerShapley calculator.
        
        Args:
            model: The neural network model
            device: Torch device (CPU/GPU)
            config: Configuration object for Shapley computation
        """
        self.model = model
        self.device = device
        self.config = config or ShapleyConfig()
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        if self.config.threshold <= 0:
            raise ValueError("Threshold must be positive")
        if self.config.max_subset_size <= 0:
            raise ValueError("Max subset size must be positive")
        if self.config.monte_carlo_samples <= 0:
            raise ValueError("Monte Carlo samples must be positive")
    
    def count_relu_layers(self) -> int:
        """
        Count the number of ReLU layers in the model.
        
        Returns:
            Number of ReLU layers (layers with 'alpha' parameters)
        """
        relu_count = 0
        for name, param in self.model.named_parameters():
            if 'alpha' in name:
                relu_count += 1
        return relu_count
    
    def find_subsets(self, nums: List[int], max_subset_size: Optional[int] = None) -> List[List[int]]:
        """
        Generate all possible subsets of the given numbers.
        
        Args:
            nums: List of layer indices
            max_subset_size: Maximum size of subsets to generate
            
        Returns:
            List of all possible subsets
        """
        if max_subset_size is None:
            max_subset_size = self.config.max_subset_size
            
        if not nums:
            return [[]]
            
        subsets = []
        start_size = max(0, len(nums) - max_subset_size)
        
        for i in range(start_size, len(nums) + 1):
            subsets.extend(itertools.combinations(nums, i))
            
        return [list(subset) for subset in subsets]
    
    def _mask_model_layers(self, model: nn.Module, mask: List[int], current_layer: int = 1) -> int:
        """
        Mask layers in the model based on the given mask.
        
        Args:
            model: The model to mask
            mask: List of layer indices to keep active
            current_layer: Current layer index (for recursive calls)
            
        Returns:
            Updated current layer index
        """
        for name, layer in model.named_children():
            if isinstance(layer, (nn.Sequential, BasicBlock_IN)):
                current_layer = self._mask_model_layers(layer, mask, current_layer)
            elif isinstance(layer, LearnableAlpha):
                if current_layer not in mask:
                    layer.alphas.data = torch.zeros_like(layer.alphas.data)
                current_layer += 1
        return current_layer
    
    def _mask_model_wrn(self, model: nn.Module, mask: List[int], current_layer: int = 1) -> int:
        """
        Mask layers in Wide ResNet architecture.
        
        Args:
            model: The WRN model to mask
            mask: List of layer indices to keep active
            current_layer: Current layer index
            
        Returns:
            Updated current layer index
        """
        for name, param in model.named_parameters():
            if "alpha" in name:
                if current_layer not in mask:
                    param.data = torch.zeros_like(param.data)
                current_layer += 1
        return current_layer
    
    def get_model_utility(self, model: nn.Module, test_loader: DataLoader, 
                         mask: List[int], base_model: nn.Module) -> float:
        """
        Calculate the utility (accuracy) of a masked model.
        
        Args:
            model: The model to evaluate
            test_loader: Data loader for testing
            mask: List of active layer indices
            base_model: Original unmasked model for comparison
            
        Returns:
            Model accuracy after masking
        """
        if not mask:
            return 0.0
            
        # Create a deep copy to avoid modifying the original
        masked_model = copy.deepcopy(base_model)
        masked_model = masked_model.to(self.device)
        
        # Apply masking based on model type
        current_layer = 1
        if self.config.model_type == ModelType.WRN:
            self._mask_model_wrn(masked_model, mask, current_layer)
        else:
            self._mask_model_layers(masked_model, mask, current_layer)
        
        # Evaluate the masked model
        return model_inference(masked_model, test_loader, self.device, display=False)
    
    def compute_single_layer_shapley(self, layer_index: int, test_loader: DataLoader,
                                   base_model: nn.Module, subsets: Optional[List[List[int]]] = None) -> float:
        """
        Compute Shapley value for a single layer.
        
        Args:
            layer_index: Index of the layer (1-based)
            test_loader: Data loader for testing
            base_model: Original unmasked model
            subsets: Pre-computed subsets (optional)
            
        Returns:
            Shapley value for the specified layer
        """
        # Input validation
        if layer_index < 1:
            raise ValueError(f"Invalid layer_index: {layer_index}")
            
        total_layers = self.count_relu_layers()
        if layer_index > total_layers:
            raise ValueError(f"Layer index {layer_index} exceeds total layers {total_layers}")
        
        # Generate subsets if not provided
        if subsets is None:
            other_layers = [i for i in range(1, total_layers + 1) if i != layer_index]
            subsets = self.find_subsets(other_layers)
        
        if not subsets:
            return 0.0
        
        shapley_value = 0.0
        
        # Process subsets in reverse order for early termination
        for mask in reversed(subsets):
            # Utility without the layer
            utility_without = self.get_model_utility(self.model, test_loader, mask, base_model)
            
            # Utility with the layer
            mask_with_layer = mask + [layer_index]
            utility_with = self.get_model_utility(self.model, test_loader, mask_with_layer, base_model)
            
            # Early termination check
            if (self.config.early_termination_threshold is not None and 
                utility_with < self.config.early_termination_threshold):
                break
            
            # Calculate marginal contribution
            marginal_contribution = utility_with - utility_without
            weight = 1.0 / comb(total_layers - 1, len(mask))
            shapley_value += marginal_contribution * weight
        
        # Normalize by total number of layers
        shapley_value = shapley_value / total_layers
        
        return shapley_value
    
    def compute_all_layer_shapley_values(self, test_loader: DataLoader,
                                       base_model: Optional[nn.Module] = None) -> List[float]:
        """
        Compute Shapley values for all layers in the model.
        
        Args:
            test_loader: Data loader for testing
            base_model: Original model (uses self.model if None)
            
        Returns:
            List of Shapley values for each layer
        """
        if base_model is None:
            base_model = self.model
            
        total_layers = self.count_relu_layers()
        if total_layers == 0:
            console.print("No ReLU layers found in model", style="bold red")
            return []
        
        console.print(f"Computing Shapley values for {total_layers} layers", style="bold green")
        
        # Pre-compute subsets for efficiency
        all_layers = list(range(1, total_layers + 1))
        subsets = self.find_subsets(all_layers)
        
        shapley_values = []
        
        # Compute Shapley value for each layer
        for layer_idx in tqdm(range(1, total_layers + 1), desc="Computing layer Shapley values"):
            try:
                # Generate subsets excluding current layer
                other_layers = [i for i in all_layers if i != layer_idx]
                layer_subsets = self.find_subsets(other_layers)
                
                sv = self.compute_single_layer_shapley(
                    layer_idx, test_loader, base_model, layer_subsets
                )
                shapley_values.append(sv)
                console.print(f"Layer {layer_idx}: {sv:.4f}")
                
            except Exception as e:
                console.print(f"Error computing Shapley value for layer {layer_idx}: {e}", 
                            style="bold red")
                shapley_values.append(0.0)
        
        return shapley_values
    
    def analyze_model(self, test_loader: DataLoader, base_model: Optional[nn.Module] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of layer importance.
        
        Args:
            test_loader: Data loader for testing
            base_model: Original model (uses self.model if None)
            
        Returns:
            Dictionary containing analysis results
        """
        if base_model is None:
            base_model = self.model
            
        # Get original model accuracy
        original_acc = model_inference(base_model, test_loader, self.device, display=False)
        
        # Compute Shapley values
        shapley_values = self.compute_all_layer_shapley_values(test_loader, base_model)
        
        # Analyze results
        total_importance = sum(shapley_values)
        layer_stats = []
        
        for i, sv in enumerate(shapley_values, 1):
            percentage = (sv / total_importance * 100) if total_importance != 0 else 0
            layer_stats.append({
                'layer_index': i,
                'shapley_value': sv,
                'importance_percentage': percentage
            })
        
        # Sort by importance
        layer_stats.sort(key=lambda x: x['shapley_value'], reverse=True)
        
        return {
            'original_accuracy': original_acc,
            'total_layers': len(shapley_values),
            'shapley_values': shapley_values,
            'layer_statistics': layer_stats,
            'total_importance': total_importance
        }


def main():
    """
    Example usage of the refactored LayerShapley class.
    Returns list of Shapley values ordered by layer index for use in SNIP function.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Layer Shapley Value Calculator')
    parser.add_argument('dataset', type=str, choices=DATASETS, help='Dataset to use')
    parser.add_argument('arch', type=str, choices=ARCHITECTURES, help='Model architecture')
    parser.add_argument('outdir', type=str, help='Output directory for results')
    parser.add_argument('savedir', type=str, help='Path to saved model checkpoint')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers')
    parser.add_argument('--batch', default=256, type=int, help='Batch size')
    parser.add_argument('--threshold', default=1e-2, type=float, help='Threshold for ReLU counting')
    parser.add_argument('--early-termination', default=5.0, type=float, 
                       help='Early termination threshold (set to None to disable)')
    parser.add_argument('--max-subset-size', default=4, type=int, 
                       help='Maximum subset size for Shapley computation')
    parser.add_argument('--model-type', default='standard', choices=['standard', 'wrn'],
                       help='Model architecture type')
    parser.add_argument('--logname', type=str, default='layer_shapley_results.txt')
    parser.add_argument('--stride', type=int, default=1, help='conv1 stride')
    
    args = parser.parse_args()
    
    # Setup output directory
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    logfilename = os.path.join(args.outdir, args.logname)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}")
    
    # Load datasets
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                           num_workers=args.workers, pin_memory=pin_memory)
    
    # Load model
    base_classifier = get_architecture(args.arch, args.dataset, device, args)
    checkpoint = torch.load(args.savedir, map_location=device)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    base_classifier.eval()
    
    # Get original accuracy
    original_acc = model_inference(base_classifier, test_loader, device, display=True)
    console.print(f"Original model accuracy: {original_acc:.4f}")
    
    # Create Shapley configuration
    config = ShapleyConfig(
        threshold=args.threshold,
        early_termination_threshold=args.early_termination,
        max_subset_size=args.max_subset_size,
        model_type=args.model_type
    )
    
    # Initialize LayerShapley calculator
    shapley_calculator = LayerShapley(base_classifier, device, config)
    
    # Count ReLU layers
    num_relu_layers = shapley_calculator.count_relu_layers()
    console.print(f"Total number of ReLU layers: {num_relu_layers}")
    
    if num_relu_layers == 0:
        console.print("No ReLU layers found. Exiting.", style="bold red")
        return []
    
    # Compute Shapley values for all layers (ordered by layer index)
    console.print("Computing Shapley values for all layers...", style="bold green")
    shapley_values = shapley_calculator.compute_all_layer_shapley_values(test_loader)
    
    # Display results
    console.print("\n" + "="*50, style="bold blue")
    console.print("LAYER IMPORTANCE ANALYSIS RESULTS", style="bold blue")
    console.print("="*50, style="bold blue")
    
    console.print(f"Original Accuracy: {original_acc:.4f}")
    console.print(f"Total Layers: {len(shapley_values)}")
    
    total_importance = sum(shapley_values)
    console.print(f"Total Shapley Importance: {total_importance:.4f}")
    
    console.print("\nLayer Shapley Values (by index):")
    console.print("-" * 40)
    
    for i, sv in enumerate(shapley_values, 1):
        percentage = (sv / total_importance * 100) if total_importance != 0 else 0
        console.print(f"Layer {i:2d}: SV={sv:8.4f} ({percentage:6.2f}%)")
    
    # Save results to file
    with open(logfilename, 'w') as f:
        f.write(f"Layer Shapley Value Analysis Results\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Architecture: {args.arch}\n")
        f.write(f"Original Accuracy: {original_acc:.4f}\n")
        f.write(f"Total Layers: {len(shapley_values)}\n")
        f.write(f"Total Shapley Importance: {total_importance:.4f}\n\n")
        
        f.write("Layer Shapley Values (ordered by layer index):\n")
        for i, sv in enumerate(shapley_values, 1):
            f.write(f"Layer {i}: {sv:.6f}\n")
    
    console.print(f"\nResults saved to: {logfilename}", style="bold green")
    
    # Return Shapley values ordered by layer index for use in SNIP function
    return shapley_values


def compute_layer_shapley_values(model: nn.Module, device: torch.device, test_loader: DataLoader,
                                model_type: str = 'standard', threshold: float = 1e-2,
                                early_termination: Optional[float] = 5.0, max_subset_size: int = 4) -> List[float]:
    """
    Compute Shapley values for all layers in a model.
    
    This function provides a direct API for computing layer Shapley values
    without going through the command line interface.
    
    Args:
        model: The neural network model
        device: Torch device (CPU/GPU)
        test_loader: Data loader for testing
        model_type: Model architecture type ('standard' or 'wrn')
        threshold: Threshold for ReLU counting
        early_termination: Early termination threshold (None to disable)
        max_subset_size: Maximum subset size for Shapley computation
        
    Returns:
        List of Shapley values ordered by layer index
    """
    # Create Shapley configuration
    config = ShapleyConfig(
        threshold=threshold,
        early_termination_threshold=early_termination,
        max_subset_size=max_subset_size,
        model_type=model_type
    )
    
    # Initialize LayerShapley calculator
    shapley_calculator = LayerShapley(model, device, config)
    
    # Count ReLU layers
    num_relu_layers = shapley_calculator.count_relu_layers()
    if num_relu_layers == 0:
        console.print("No ReLU layers found. Returning empty list.", style="bold red")
        return []
    
    # Compute Shapley values for all layers (ordered by layer index)
    console.print(f"Computing Shapley values for {num_relu_layers} layers...", style="bold green")
    shapley_values = shapley_calculator.compute_all_layer_shapley_values(test_loader)
    
    return shapley_values


if __name__ == "__main__":
    main()