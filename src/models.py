#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyArrow
from tqdm import tqdm
import embeddings as E
from box import Box
from util import transitive_roles_dict
from scipy.stats import spearmanr
import numpy as np
from typing import Dict, List, Tuple

def Identity(x):
    return x

class KGReasoning(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma, alpha,
                 geo, test_batch_size=1, box_mode=None,
                 use_cuda=False, query_name_dict=None, beta_mode=None, transitive_ids=None, inverse_ids=None, with_answer_embedding=False):
        super(KGReasoning, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.geo = geo
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda() if self.use_cuda else torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1) # used in test_step
        self.query_name_dict = query_name_dict

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        
        self.transitive_ids = nn.Parameter(torch.LongTensor(transitive_ids), requires_grad=False)
        self.inverse_ids = nn.Parameter(torch.LongTensor(inverse_ids), requires_grad=False)
        
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )

        self.alpha = nn.Parameter(
            torch.Tensor([alpha]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )

        
        
        self.center_embedding = self.init_embedding(nentity, self.entity_dim)
        self.offset_embedding = self.init_embedding(nentity, self.entity_dim)

        self.with_answer_embedding = with_answer_embedding

        if self.with_answer_embedding:
            self.answer_embedding = self.init_embedding(nentity, self.entity_dim)

        self.center_mul = self.init_embedding(nrelation, self.relation_dim)
        self.center_add = self.init_embedding(nrelation, self.relation_dim)
        self.offset_mul = self.init_embedding(nrelation, self.relation_dim)
        self.offset_add = self.init_embedding(nrelation, self.relation_dim)

    def init_embedding(self, num_embeddings, dimension):
        embedding = nn.Embedding(num_embeddings, dimension)
        nn.init.uniform_(
            tensor=embedding.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        return embedding
    #####################################


    def evaluate_sequence_monotonicity(self, seq_indices: List[str], embeddings: torch.Tensor) -> Tuple[float, List[float]]:
        """
        Calculate monotonicity score for a single sequence using our new approach.

        Args:
            seq_indices: List of indices in the sequence
            embeddings: Tensor containing embedding values

        Returns:
            Tuple of (sequence_score, list_of_element_scores)
        """
        n = len(seq_indices)
        element_scores = []

        # Convert string indices to integers for embedding lookup
        indices = [int(idx) for idx in seq_indices]

        # Get embedding values for this sequence
        seq_embeddings = torch.tensor([embeddings[idx].item() for idx in indices])

        for i in range(n):
            # Count elements before current element with lower embedding values
            correct_before = 0
            total_before = i
            if total_before > 0:
                correct_before = torch.sum(seq_embeddings[:i] < seq_embeddings[i]).item()

            # Count elements after current element with higher embedding values
            correct_after = 0
            total_after = n - i - 1
            if total_after > 0:
                correct_after = torch.sum(seq_embeddings[i+1:] > seq_embeddings[i]).item()

            # Calculate score for this element
            total_pairs = total_before + total_after
            if total_pairs > 0:
                element_score = (correct_before + correct_after) / total_pairs
            else:
                element_score = 1.0  # If no pairs to compare

            element_scores.append(element_score)

        # Average score for the sequence
        sequence_score = sum(element_scores) / n if n > 0 else 0.0

        return sequence_score, element_scores


    def count_pairwise_violations(self, seq_indices: List[str], embeddings: torch.Tensor) -> Tuple[int, int]:
        """
        Count pairwise violations in a sequence.

        Args:
            seq_indices: List of indices in the sequence
            embeddings: Tensor containing embedding values

        Returns:
            Tuple of (violation_count, total_pairs)
        """
        # Convert string indices to integers for embedding lookup
        indices = [int(idx) for idx in seq_indices]

        # Get embedding values for this sequence
        seq_embeddings = torch.tensor([embeddings[idx].item() for idx in indices])

        violation_count = 0
        total_pairs = 0

        # Check each pair for violations
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                # Expected: earlier elements should have lower embedding values
                total_pairs += 1
                if seq_embeddings[i] >= seq_embeddings[j]:
                    violation_count += 1

        return violation_count, total_pairs



    def compute_monotonicity_metrics(self, args, weighting_method='linear'):
        """
        Compute monotonicity metrics for transitive relations, with length-based weighting.

        Args:
            args: Arguments object with data_path
            weighting_method: Method to weight sequences by length ('linear', 'quadratic', 'exponential')

        Returns:
            Tuple of dictionaries with weighted average monotonicity scores, 
            violation percentages, and Spearman scores.
        """
        transitive_roles = transitive_roles_dict["WN18RR-QA"]  # Using the same transitive roles
        transitive_ids = [0, 1,2,3,12,13]  # Using the provided transitive ids

        # Initialize result dictionaries
        monotonicity_scores = {i: list() for i in transitive_ids}
        violation_percentages = {i: list() for i in transitive_ids}
        spearman_scores = {i: list() for i in transitive_ids}  # Keep Spearman for comparison
        sequence_lengths = {i: list() for i in transitive_ids}  # Track sequence lengths for weighting

        for id in transitive_ids:
            # Extract embeddings (using the same logic as in original code)
            center = self.center_embedding.weight[:, id].detach().cpu()
            offset = torch.abs(self.offset_embedding.weight[:, id]).detach().cpu()
            upper = center + offset
            embeddings = upper  # Or any other embedding you want to evaluate

            # Read chains from file (same as original)
            filename = os.path.join(args.data_path, f"chains_{id}.txt")
            with open(filename) as f:
                chains = f.readlines()
                chains = sorted([c.strip().split(",") for c in chains], key=lambda x: (len(x), x))
                chains = [c for c in chains if len(c) > 2]  # Only use chains with more than 2 elements

            # Process each chain (sequence)
            for seq in chains:
                seq_len = len(seq)
                sequence_lengths[id].append(seq_len)

                # Calculate our new monotonicity score
                seq_score, element_scores = self.evaluate_sequence_monotonicity(seq, -embeddings)
                monotonicity_scores[id].append(seq_score)

                # Calculate violation percentage
                violation_count, total_pairs = self.count_pairwise_violations(seq, -embeddings)
                violation_pct = violation_count / total_pairs if total_pairs > 0 else 0
                violation_percentages[id].append(violation_pct)

                # Also keep the original Spearman calculation for comparison
                values = [-embeddings[int(e)].item() for e in seq]
                positions = list(range(len(seq)))
                rho, _ = spearmanr(positions, values)
                spearman_scores[id].append(rho)

        # Compute weighted averages
        avg_monotonicity = {}
        avg_violation_pct = {}
        avg_spearman = {}

        for id in transitive_ids:
            # Calculate weights based on sequence lengths
            weights = self.compute_length_weights(sequence_lengths[id], method=weighting_method)

            # Apply weights to each metric
            avg_monotonicity[id] = np.average(monotonicity_scores[id], weights=weights)
            avg_violation_pct[id] = np.average(violation_percentages[id], weights=weights)
            avg_spearman[id] = np.average(spearman_scores[id], weights=weights)

        # Create a consolidated results dictionary
        results = {
            "monotonicity_scores": avg_monotonicity,
            "violation_percentages": avg_violation_pct,
            "spearman_scores": avg_spearman,
            "raw_monotonicity_scores": monotonicity_scores,
            "raw_violation_percentages": violation_percentages,
            "raw_spearman_scores": spearman_scores,
            "sequence_lengths": sequence_lengths,
            "weighting_method": weighting_method
        }

        print(f"Monotonicity:{avg_monotonicity}")
        print(f"Violation %:{avg_violation_pct}")
        print(f"Spearman Sc:{avg_spearman}")
        
        return results


    def compute_length_weights(self,sequence_lengths, method='linear'):
        """
        Compute weights based on sequence lengths such that the weights sum to 1.
        Longer sequences receive higher weights.

        Args:
            sequence_lengths: List of sequence lengths
            method: Method to compute weights ('linear', 'quadratic', 'exponential')

        Returns:
            Array of weights that sum to 1
        """
        lengths = np.array(sequence_lengths)

        # Normalize lengths to the minimum length (typically 3)
        min_length = np.min(lengths)
        normalized_lengths = lengths / min_length

        if method == 'linear':
            # Linear weighting: weight = length
            raw_weights = normalized_lengths
        elif method == 'quadratic':
            # Quadratic weighting: weight = length^2
            raw_weights = normalized_lengths ** 2
        elif method == 'exponential':
            # Exponential weighting: weight = e^(length/min_length - 1)
            raw_weights = np.exp(normalized_lengths - 1)
        else:
            raise ValueError(f"Unknown weighting method: {method}")

        # Normalize weights to sum to 1
        normalized_weights = raw_weights / np.sum(raw_weights)

        return normalized_weights
                        
    #####################################


    def plot_chain_progress(self, args):
        """
        Plot chains as progress lines, where vertical segments indicate preserved order
        and horizontal segments indicate violations.

        Starting at (0,0) for each chain:
        - Move upward (vertical line) when order is preserved
        - Move rightward (horizontal line) when order is violated

        Args:
            args: Arguments object with data_path and save_path
        """
        # Get monotonicity metrics
        results = compute_monotonicity_metrics(self, args)

        transitive_ids = [0, 1, 2,3,12,13]

        for id in transitive_ids:
            # Extract embeddings
            embeddings = self.center_embedding.weight[:, id].detach().cpu()

            # Read chains from file
            filename = os.path.join(args.data_path, f"chains_{id}.txt")
            with open(filename) as f:
                chains = f.readlines()
                chains = sorted([c.strip().split(",") for c in chains], key=lambda x: (len(x), x))
                chains = [c for c in chains if len(c) > 2]

            # Sort chains by length for better visualization
            chains = sorted(chains, key=len, reverse=True)

            # Take only the first 30 chains for clarity
            display_chains = chains[:30]

            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(16, 10))
            ax_progress = axes[0]
            ax_lines = axes[1]

            # Track the max progress values for axis limits
            max_x = 0
            max_y = 0

            # Colors
            preserved_color = 'green'
            violated_color = 'red'

            # Plot progress lines
            for chain_idx, seq in enumerate(display_chains):
                values = [-embeddings[int(e)].item() for e in seq]

                # Start position for this chain
                x, y = 0, 0
                path_x = [x]
                path_y = [y]

                # Track progress through the sequence
                for i in range(len(values) - 1):
                    if values[i] <= values[i + 1]:
                        # Order preserved - move upward
                        y += 1
                        path_x.append(x)
                        path_y.append(y)
                    else:
                        # Order violated - move rightward
                        x += 1
                        path_x.append(x)
                        path_y.append(y)

                # Update max values
                max_x = max(max_x, max(path_x))
                max_y = max(max_y, max(path_y))

                # Plot this chain's path
                monotonicity_ratio = y / (len(values) - 1)
                color = plt.cm.viridis(monotonicity_ratio)  # Color based on monotonicity ratio
                ax_progress.plot(path_x, path_y, '-', linewidth=1.5, alpha=0.7, color=color)

                # Add marker for endpoint
                ax_progress.scatter(path_x[-1], path_y[-1], s=30, color=color)

                # Optional: Add chain label for the endpoint
                if chain_idx < 10:  # Only label first few chains
                    ax_progress.annotate(f"{chain_idx+1}", 
                                       (path_x[-1], path_y[-1]),
                                       xytext=(5, 0),
                                       textcoords='offset points',
                                       fontsize=8)

            # Draw ideal path (perfect monotonicity) in light gray
            max_seq_len = max(len(seq) for seq in display_chains)
            ideal_x = [0]
            ideal_y = list(range(max_seq_len))
            ax_progress.plot(ideal_x, ideal_y, '--', color='gray', alpha=0.5, linewidth=1)

            # Set axis limits with some padding
            ax_progress.set_xlim(-0.5, max_x + 2)
            ax_progress.set_ylim(-0.5, max(max_y + 2, max_seq_len))

            # Add grid for reference
            ax_progress.grid(True, linestyle='--', alpha=0.3)

            # Add labels and title
            ax_progress.set_xlabel("Number of Order Violations")
            ax_progress.set_ylabel("Number of Order Preservations")
            ax_progress.set_title(f"Monotonicity Progress - ID {id}")

            # Add diagonal reference lines to show % of monotonicity
            for ratio in [0.25, 0.5, 0.75]:
                max_val = max(max_x, max_y)
                x_vals = np.linspace(0, max_val, 100)
                y_vals = ratio / (1 - ratio) * x_vals if ratio < 1 else x_vals
                ax_progress.plot(x_vals, y_vals, ':', color='gray', alpha=0.5)
                # Add label for ratio line
                midpoint = max_val // 2
                ax_progress.annotate(f"{ratio:.0%}", 
                                   (midpoint, ratio / (1 - ratio) * midpoint if ratio < 1 else midpoint),
                                   xytext=(0, 5),
                                   textcoords='offset points',
                                   fontsize=8,
                                   color='gray')

            # Create second plot: segment visualization
            for idx, seq in enumerate(display_chains):
                values = [-embeddings[int(e)].item() for e in seq]

                y_gap = 1  # Vertical space between sequences
                y = idx * y_gap  # Sequence row

                for i in range(len(values) - 1):
                    x_start = i
                    x_end = i + 1

                    # Determine segment color based on order preservation
                    if values[i] <= values[i + 1]:
                        color = preserved_color  # Order preserved
                    else:
                        color = violated_color  # Order violated

                    # Draw line segment
                    ax_lines.plot([x_start, x_end], [y, y], color=color, linewidth=2)

                    # Optional: Add dots at each position
                    ax_lines.scatter(x_start, y, color='black', s=10, zorder=10)

                # Add dot for last position
                ax_lines.scatter(len(values)-1, y, color='black', s=10, zorder=10)

                # Optional: Label with length and violation count
                violation_count = sum(1 for i in range(len(values)-1) if values[i] > values[i+1])
                preservation_count = (len(values) - 1) - violation_count
                monotonicity_ratio = preservation_count / (len(values) - 1)

                label = f"{len(seq)} ({violation_count})"
                ax_lines.text(-1, y, label, va='center', ha='right', fontsize=8)

                # Add monotonicity percentage
                ax_lines.text(len(values) + 0.5, y, f"{monotonicity_ratio:.0%}", va='center', ha='left', 
                            fontsize=8, color='blue')

            # Formatting for line segments plot
            ax_lines.set_ylim(-1, len(display_chains) * y_gap)
            max_length = max(len(seq) for seq in display_chains)
            ax_lines.set_xlim(-2, max_length + 3)
            ax_lines.set_yticks([])
            ax_lines.set_xlabel("Position in Sequence")
            ax_lines.set_title(f"Segment Visualization - ID {id}\nGreen: Order Preserved, Red: Order Violated")

            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color=preserved_color, lw=2, label='Order Preserved'),
                Line2D([0], [0], color=violated_color, lw=2, label='Order Violated')
            ]
            ax_lines.legend(handles=legend_elements, loc='upper right')

            # Add labels for axes
            ax_lines.text(-1.5, len(display_chains) * y_gap / 2, "Sequences", 
                        rotation=90, va='center', ha='center', fontsize=12)

            # Overall formatting
            plt.tight_layout()

            # Add a colorbar for the progress plot
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                    norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax_progress)
            cbar.set_label('Monotonicity Ratio')

            # Add overall title
            fig.suptitle(f"Chain Monotonicity Visualization for Transitive ID {id}", fontsize=16, y=0.98)
            plt.subplots_adjust(top=0.9)

            # Save figure
            outfilename = os.path.join(args.save_path, f"chains_monotonicity_plot_{id}.png")
            plt.savefig(outfilename, dpi=300, bbox_inches='tight')
            plt.close()


    def plot_monotonicity_heatmap(self, args):
        """
        Create a heatmap visualization of monotonicity across chains of different lengths.

        Args:
            args: Arguments object with data_path and save_path
        """
        transitive_ids = [0, 1, 2, 3, 12, 13]

        for id in transitive_ids:
            # Extract embeddings
            embeddings = self.center_embedding.weight[:, id].detach().cpu()

            # Read chains from file
            filename = os.path.join(args.data_path, f"chains_{id}.txt")
            with open(filename) as f:
                chains = f.readlines()
                chains = sorted([c.strip().split(",") for c in chains], key=lambda x: (len(x), x))
                chains = [c for c in chains if len(c) > 2]

            # Collect data by length
            length_data = {}
            for seq in chains:
                length = len(seq)
                if length not in length_data:
                    length_data[length] = []

                # Calculate monotonicity for this sequence
                values = [-embeddings[int(e)].item() for e in seq]
                violation_count = sum(1 for i in range(len(values)-1) if values[i] > values[i+1])
                monotonicity_ratio = 1 - (violation_count / (len(values) - 1))

                length_data[length].append(monotonicity_ratio)

            # Prepare heatmap data
            lengths = sorted(length_data.keys())

            # Create figure
            plt.figure(figsize=(12, 8))

            # Create boxplot for monotonicity by length
            box_data = [length_data[l] for l in lengths]
            plt.boxplot(box_data, labels=lengths)

            # Add jittered points for individual chains
            for i, length in enumerate(lengths):
                x = np.random.normal(i+1, 0.05, size=len(length_data[length]))
                plt.scatter(x, length_data[length], alpha=0.4, s=20, c='blue')

            # Add mean line
            means = [np.mean(length_data[l]) for l in lengths]
            plt.plot(range(1, len(lengths)+1), means, 'r-', linewidth=2, label='Mean')

            # Formatting
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.xlabel('Sequence Length')
            plt.ylabel('Monotonicity Ratio')
            plt.title(f'Monotonicity by Sequence Length - ID {id}')
            plt.ylim(-0.05, 1.05)
            plt.legend()

            # Add reference line at y=0.5
            plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)

            # Annotate means
            for i, mean in enumerate(means):
                plt.annotate(f'{mean:.2f}', 
                           (i+1, mean), 
                           xytext=(0, 5),
                           textcoords='offset points',
                           ha='center')

            # Save figure
            outfilename = os.path.join(args.save_path, f"monotonicity_by_length_{id}.png")
            plt.savefig(outfilename, dpi=300, bbox_inches='tight')
            plt.close()


    def plot_stacked_progression(self, args):
        """
        Create a stacked area chart showing the progression of preserved vs. violated
        ordering across sequence positions.

        Args:
            args: Arguments object with data_path and save_path
        """
        transitive_ids = [0, 1, 2, 3, 12, 13]

        for id in transitive_ids:
            # Extract embeddings
            embeddings = self.center_embedding.weight[:, id].detach().cpu()

            # Read chains from file
            filename = os.path.join(args.data_path, f"chains_{id}.txt")
            with open(filename) as f:
                chains = f.readlines()
                chains = sorted([c.strip().split(",") for c in chains], key=lambda x: (len(x), x))
                chains = [c for c in chains if len(c) > 2]

            # Group by length
            chains_by_length = {}
            for seq in chains:
                length = len(seq)
                if length not in chains_by_length:
                    chains_by_length[length] = []
                chains_by_length[length].append(seq)

            # Create a progression analysis by position
            max_length = max(len(seq) for seq in chains)
            position_data = {
                'preserved': np.zeros(max_length - 1),
                'violated': np.zeros(max_length - 1),
                'total': np.zeros(max_length - 1)
            }

            # Analyze each chain
            for seq in chains:
                values = [-embeddings[int(e)].item() for e in seq]

                for i in range(len(values) - 1):
                    position_data['total'][i] += 1
                    if values[i] <= values[i + 1]:
                        position_data['preserved'][i] += 1
                    else:
                        position_data['violated'][i] += 1

            # Calculate percentages
            preserved_pct = position_data['preserved'] / position_data['total'] * 100
            violated_pct = position_data['violated'] / position_data['total'] * 100

            # Create stacked area chart
            plt.figure(figsize=(12, 6))

            x = np.arange(max_length - 1)
            plt.stackplot(x, 
                        [preserved_pct, violated_pct], 
                        labels=['Preserved', 'Violated'],
                        colors=['green', 'red'],
                        alpha=0.7)

            # Add count line
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax2.plot(x, position_data['total'], 'k--', linewidth=1.5, label='Chain Count')
            ax2.set_ylabel('Number of Chains')

            # Formatting
            plt.grid(True, linestyle='--', alpha=0.3)
            ax1.set_xlabel('Position in Sequence')
            ax1.set_ylabel('Percentage')
            plt.title(f'Order Preservation by Position - ID {id}')
            ax1.set_xlim(0, max_length - 2)
            ax1.set_ylim(0, 100)

            # Add x-ticks
            plt.xticks(range(max_length - 1))
            ax1.set_xticklabels([f'{i}-{i+1}' for i in range(max_length - 1)])

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

            # Add annotations for percentages
            for i in range(len(preserved_pct)):
                if position_data['total'][i] > max(position_data['total']) * 0.5:  # Only annotate positions with enough data
                    plt.annotate(f'{preserved_pct[i]:.0f}%', 
                               (i, preserved_pct[i]/2), 
                               ha='center', 
                               color='white', 
                               fontweight='bold')

                    plt.annotate(f'{violated_pct[i]:.0f}%', 
                               (i, preserved_pct[i] + violated_pct[i]/2), 
                               ha='center', 
                               color='white', 
                               fontweight='bold')

            # Save figure
            outfilename = os.path.join(args.save_path, f"ordering_by_position_{id}.png")
            plt.savefig(outfilename, dpi=300, bbox_inches='tight')
            plt.close()



    def compute_spearman_and_violations(self, args):
        transitive_roles = transitive_roles_dict["WN18RR-QA"]
        transitive_ids = [0,1,2,3,12,13] #self.transitive_ids.cpu().numpy().tolist()

        spearman_scores = {i: list() for i in transitive_ids}
        violation_counts = {i: list() for i in transitive_ids}

        for id in transitive_ids:
            # center = self.center_embedding.weight[:, id].detach().cpu()
            # offset = torch.abs(self.offset_embedding.weight[:, id]).detach().cpu()
            # ;upper = center + offset
            # embeddings = upper
            embeddings = self.center_embedding.weight[:, id].detach().cpu()
            filename = os.path.join(args.data_path, f"chains_{id}.txt")
            with open(filename) as f:
                chains = f.readlines()
                chains = sorted([c.split(",") for c in chains], key=lambda x: (len(x), x))
                # chains = [c[::-1] for c in chains]
                chains = [c for c in chains if len(c)>2]

            for seq in chains:
                values = [-embeddings[int(e)] for e in seq]
                positions = list(range(len(seq)))

                # Spearman's rho
                rho, _ = spearmanr(positions, values)
                spearman_scores[id].append(rho)

                # Count order violations (for increasing order)
                violations = sum(1 for i in range(len(values)-1) if values[i] > values[i+1])
                violation_counts[id].append(violations)


        # Summary
        avg_spearman = {id: np.mean(scores) for id, scores in spearman_scores.items()}
        avg_violations = {id: np.mean(violations) for id, violations in violation_counts.items()}
        return avg_spearman, avg_violations
        # print("Spearman's rho and violations:")
        # for id in transitive_ids:
            # print(f"ID {id}:")
            # print(f"  Average Spearman's rho: {avg_spearman[id]:.2f}")
            # print(f"  Average violations: {avg_violations[id]:.2f}")


    def plot_chain_arrows(self, args):
        spearman_scores, violation_counts = self.compute_spearman_and_violations(args)
        print(spearman_scores)
        print(violation_counts)
        transitive_roles = transitive_roles_dict["WN18RR-QA"]
        transitive_ids = [0,1,2,3,12,13] #self.transitive_ids.cpu().numpy().tolist()

        for id in transitive_ids:
            right_color = 'green'
            left_color = 'red'

            # if spearman_scores[id] < 0:
                # right_color = 'red'
                # left_color = 'green'
            # else:
                # right_color = 'green'
                # left_color = 'red'
            # center = self.center_embedding.weight[:, id].detach().cpu()
            # offset = torch.abs(self.offset_embedding.weight[:, id]).detach().cpu()
            # upper = center + offset
            # embeddings = upper
            embeddings = self.center_embedding.weight[:, id].detach().cpu()

            filename = os.path.join(args.data_path, f"chains_{id}.txt")
            with open(filename) as f:
                chains = f.readlines()
                chains = sorted([c.split(",") for c in chains], key=lambda x: (len(x), x))
                # chains = [c[::-1] for c in chains]
                chains = [c for c in chains if len(c)>2]
            max_length = max(len(seq) for seq in chains)

            fig, ax = plt.subplots(figsize=(12, 10))
            for idx, seq in enumerate(chains):
                values = [-embeddings[int(e)] for e in seq]
                y_gap = 1  # Vertical space between sequences
                y = idx * y_gap  # Sequence row

                for i in range(len(values) - 1):
                    x_start = i
                    x_end = i + 1


                    if values[i] <= values[i + 1]:
                        color = right_color  # Order preserved
                        direction = 0.8
                    else:
                        color = left_color  # Order violated
                        direction = -0.8

                    # Draw arrow
                    ax.add_patch(FancyArrow(x_start, y, 0.9, 0, width=0.05, color=color))

                # Optional: Label sequences
                # ax.text(-1, y, f"Seq {idx+1}", va='center', ha='right', fontsize=8)

            # Formatting
            ax.set_ylim(-1, len(chains) * y_gap)
            ax.set_xlim(-2, max_length + 1)
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticks(range(max_length))
            ax.set_xlabel("Position in Sequence")
            ax.set_title("Order Preservation Visualization (Green: Preserved, Red: Violated)")

            plt.tight_layout()
            outfilename = os.path.join(args.save_path, f"chains_arrows_plot_{id}.png")
            plt.savefig(outfilename, dpi=300)
            plt.close()


    def plot_chains(self, args):
        transitive_roles = transitive_roles_dict["WN18RR-QA"]


        transitive_ids = [0,1,2,3,12,13] #self.transitive_ids.cpu().numpy().tolist()

        for id in transitive_ids:
            embeddings = self.answer_embedding.weight[:, id].detach().cpu()

            filename = os.path.join(args.data_path, f"chains_{id}.txt")
            with open(filename) as f:
                chains = f.readlines()
                chains = sorted([c.split(",") for c in chains], key=lambda x: (len(x), x))
                # chains = [c for c in chains if len(c)<=10][-10:]

            good_indices = list()
            good_chains = list()

            for i, c in enumerate(chains):
                if len(chains) < 3:
                    continue
                elements = [int(e) for e in c]
                elements_embed = embeddings[elements].cpu().numpy().tolist()
                sorted_embeds = sorted(elements_embed)
                if elements_embed == sorted_embeds:
                    good_indices.append(i)
                    good_chains.append(c)

            print(len(good_indices)/len(chains))
            print(good_indices[-10:])
            chains = good_chains[-10:]

            fig, ax = plt.subplots()

            for i, chain in enumerate(chains):

                elements = [int(e) for e in chain]
                y_pos = i*10 +1
                y = [y_pos]*len(elements)
                elements_embed = embeddings[elements]
                ax.scatter(elements_embed, y, zorder=2)
                k=0
                for xi, yi, label in zip(elements_embed, y, elements):
                    if k % 2 == 0:
                        position_y = "top"
                    else:
                        position_y = "bottom"
                    k += 1
                    ax.text(xi, yi + 0.05, str(label), ha='center', va=position_y, fontsize=8, zorder=4)

                for j in range(1, len(elements)):
                    x0 = embeddings[elements[j-1]].item()
                    x1 = embeddings[elements[j]].item()
                    arrow = FancyArrowPatch(
                        (x0, y_pos), (x1, y_pos),
                        connectionstyle="arc3,rad=0.1",  # positive for upward curve, negative for downward
                        arrowstyle='-|>,head_length=2,head_width=1',

                        # arrowstyle='->',
                        color='blue',
                        linewidth=0.5,
                        zorder=3

                    )
                    ax.add_patch(arrow)                     

            ax.set_yticks([])
            ax.set_yticklabels([])
            plt.title(f"{transitive_roles[id]}")

            outfilename = os.path.join(args.save_path, f"chains_plot_{id}.png")
            plt.savefig(outfilename, dpi=300)
            plt.close()

    
    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict, transitive=False):
        return self.forward_box(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict, transitive=transitive)

    def get_box_data(self):
        return self.center_embedding, self.offset_embedding

    def get_role_data(self):
        return self.center_mul, self.center_add, self.offset_mul, self.offset_add

            
    def embedding_1p(self, data, transitive):
        return E.embedding_1p(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_2p(self, data, transitive):
        return E.embedding_2p(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_3p(self, data, transitive):
        return E.embedding_3p(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_2i(self, data, transitive):
        return E.embedding_2i(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_3i(self, data, transitive):
        return E.embedding_3i(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_2in(self, data, transitive):
        return E.embedding_2in(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)
    
    def embedding_3in(self, data, transitive):
        return E.embedding_3in(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_pi(self, data, transitive):
        return E.embedding_pi(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_ip(self, data, transitive):
        return E.embedding_ip(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_inp(self, data, transitive):
        return E.embedding_inp(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)
         
    def embedding_pin(self, data, transitive):
        return E.embedding_pin(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_pni(self, data, transitive):
        return E.embedding_pni(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)
    
    def get_embedding_fn(self, task_name):
        """
        This chooses the corresponding embedding fuction given the name of the task.
        """

        return {
            "1p": self.embedding_1p,
            "2p": self.embedding_2p,
            "3p": self.embedding_3p,
            "2i": self.embedding_2i,
            "3i": self.embedding_3i,
            "2in": self.embedding_2in,
            "3in": self.embedding_3in,
            "pi": self.embedding_pi,
            "ip": self.embedding_ip,
            "inp": self.embedding_inp,
            "pin": self.embedding_pin,
            "pni": self.embedding_pni
        }[task_name]
    
    
    def embed_query_box(self, queries, query_type, transitive):
        '''
        Iterative embed a batch of queries with same structure using Query2box
        queries: a flattened batch of queries
        '''
        
        embedding_fn = self.get_embedding_fn(query_type)
        return embedding_fn(queries, transitive)


    def transform_union_query(self, queries, query_structure):
        '''
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        '''
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1] # remove union -1
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1), torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0]*2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            return ('e', ('r',))
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ('e', ('r', 'r'))

    def cal_membership_logit(self, entity_embedding, box_embedding):
        return Box.box_inclusion_score(box_embedding, entity_embedding, self.alpha)

    def cal_transitive_relation_logit(self, transitive_ids, inverse_ids):
        inverse_mask = torch.isin(transitive_ids, inverse_ids)
        
        cen_mul = self.center_mul(transitive_ids)
        cen_add = self.center_add(transitive_ids)
        off_mul = self.offset_mul(transitive_ids)
        off_add = self.offset_add(transitive_ids)

        cen_mul_loss = torch.linalg.norm(cen_mul - 1, ord=1) + torch.linalg.norm(cen_mul -1, dim=-1, ord=1)
        cen_add_loss = torch.linalg.norm(cen_add, ord=1)
        off_mul_loss = torch.linalg.norm(off_mul - 1, ord=1) + torch.linalg.norm(off_mul -1, dim=-1, ord=1)
        off_add_loss = torch.linalg.norm(off_add, ord=1)

        loss = cen_mul_loss + cen_add_loss + off_mul_loss + off_add_loss
        return loss

        
        projection_dims = torch.arange(len(transitive_ids))
        n, dim = cen_mul.shape
        mask = torch.ones((n, dim), dtype=torch.bool)
        mask[torch.arange(n), projection_dims] = False

        cen_mul_non_trans = cen_mul[mask].reshape(n, dim - 1)
        cen_mul_trans = cen_mul[torch.arange(n), projection_dims]
        cen_add_non_trans = cen_add[mask].reshape(n, dim - 1)
        cen_add_trans = cen_add[torch.arange(n), projection_dims]
        off_mul_non_trans = off_mul[mask].reshape(n, dim - 1)
        off_mul_trans = off_mul[torch.arange(n), projection_dims]
        off_add_non_trans = off_add[mask].reshape(n, dim - 1)
        off_add_trans = off_add[torch.arange(n), projection_dims]

        cen_mul_loss = torch.linalg.norm(cen_mul_non_trans - 1, ord=1) + torch.linalg.norm(cen_mul_trans-1, dim=-1, ord=1)
        cen_add_loss = torch.linalg.norm(cen_add_non_trans, ord=1)
        off_mul_loss = torch.linalg.norm(off_mul_non_trans - 1, ord=1) + torch.linalg.norm(off_mul_trans-1, dim=-1, ord=1)
        off_add_loss = torch.linalg.norm(off_add_non_trans, ord=1)

        loss = cen_mul_loss + cen_add_loss + off_mul_loss + off_add_loss
        return loss

    def cal_logit_box(self, entity_embedding, box_embedding, trans_inv, trans_not_inv, projection_dims, transitive=False, negative=False):
        if transitive:
            logit = Box.box_composed_score_with_projection(box_embedding, entity_embedding, self.alpha, trans_inv, trans_not_inv, projection_dims, negative=negative, transitive=transitive)
        else:
            logit = Box.box_inclusion_score(box_embedding, entity_embedding, self.alpha, negative=negative)
        
        return self.gamma - logit

    def forward_box(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict, transitive=False):
        all_boxes, all_idxs, all_trans_masks, all_inv_masks, all_projection_dims = [], [], [], [], []
        all_union_boxes, all_union_idxs, all_union_trans_masks, all_union_inv_masks, all_union_projection_dims = [], [], [], [], []
        for query_structure in batch_queries_dict:
            query_type = self.query_name_dict[query_structure]
            if 'u' in self.query_name_dict[query_structure]:
                query_type = self.query_name_dict[self.transform_union_structure(query_structure)]
                boxes, inv_mask, trans_mask, projection_dims = self.embed_query_box(self.transform_union_query(batch_queries_dict[query_structure], query_structure), query_type, transitive)
                all_union_boxes.append(boxes)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_trans_masks.append(trans_mask)
                all_union_inv_masks.append(inv_mask)
                all_union_projection_dims.append(projection_dims)
            else:
                boxes, inv_mask, trans_mask, projection_dims = self.embed_query_box(batch_queries_dict[query_structure], query_type, transitive)
                all_boxes.append(boxes)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_trans_masks.append(trans_mask)
                all_inv_masks.append(inv_mask)
                all_projection_dims.append(projection_dims)

        if len(all_boxes) > 0:
            all_boxes = Box.cat(all_boxes, dim=0)
            all_boxes.center = all_boxes.center.unsqueeze(1)
            all_boxes.offset = all_boxes.offset.unsqueeze(1)
            all_trans_masks = torch.cat(all_trans_masks, dim=0)
            all_inv_masks = torch.cat(all_inv_masks, dim=0)
            all_projection_dims = torch.cat(all_projection_dims, dim=0).long()
        if len(all_union_boxes) > 0:
            all_union_boxes = Box.cat(all_union_boxes, dim=0)
            all_union_boxes.center = all_union_boxes.center.unsqueeze(1)
            all_union_boxes.offset = all_union_boxes.offset.unsqueeze(1)
            all_union_trans_masks = torch.cat(all_union_trans_masks, dim=0)
            all_union_inv_masks = torch.cat(all_union_inv_masks, dim=0)
            all_union_projection_dims = torch.cat(all_union_projection_dims, dim=0).long()
        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_boxes) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                if self.with_answer_embedding:
                    positive_center_embedding = self.answer_embedding(positive_sample_regular).unsqueeze(1)
                else:
                    positive_center_embedding = self.center_embedding(positive_sample_regular).unsqueeze(1)
                positive_box = Box(positive_center_embedding, as_point=True)
                positive_logit = self.cal_logit_box(positive_box, all_boxes, all_inv_masks, all_trans_masks, all_projection_dims, transitive=transitive)
            else:
                positive_logit = torch.Tensor([]).to(self.center_embedding.weight.device)
                
            if len(all_union_boxes) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                if self.with_answer_embedding:
                    positive_center_embedding = self.answer_embedding(positive_sample_union).unsqueeze(1).unsqueeze(1)
                else:
                    positive_center_embedding = self.center_embedding(positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_box = Box(positive_center_embedding, as_point=True)
                positive_union_logit = self.cal_logit_box(positive_box, all_union_boxes, all_union_inv_masks, all_union_trans_masks, all_union_projection_dims, transitive=transitive)
                positive_union_logit = positive_union_logit.unsqueeze(1).view(positive_union_logit.shape[0]//2, 2)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.center_embedding.weight.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None
            
        if type(negative_sample) != type(None):
            if len(all_boxes) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                if self.with_answer_embedding:
                    negative_center_embedding = self.answer_embedding(negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                else:
                    negative_center_embedding = self.center_embedding(negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                negative_box = Box(negative_center_embedding, as_point=True)
                negative_logit = self.cal_logit_box(negative_box, all_boxes, all_inv_masks, all_trans_masks, all_projection_dims, negative=True, transitive=transitive)
            else:
                negative_logit = torch.Tensor([]).to(self.center_embedding.weight.device)

            if len(all_union_boxes) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                if self.with_answer_embedding:
                    negative_center_embedding = self.answer_embedding(negative_sample_union.view(-1)).view(batch_size, negative_size, -1).repeat(2, 1, 1)
                else:
                    negative_center_embedding = self.center_embedding(negative_sample_union.view(-1)).view(batch_size, negative_size, -1).repeat(2, 1, 1)
                negative_box = Box(negative_center_embedding, as_point=True)
                negative_union_logit = self.cal_logit_box(negative_box, all_union_boxes, all_union_inv_masks, all_union_trans_masks, all_union_projection_dims, negative=True, transitive=transitive)
                negative_union_logit = negative_union_logit.unsqueeze(1).view(negative_union_logit.shape[0]//2, 2, -1)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.center_embedding.weight.device)

            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        all_query_boxes = Box(self.center_embedding.weight, self.offset_embedding.weight)
        if self.with_answer_embedding:
            all_answer_boxes = Box(self.answer_embedding.weight, as_point=True)
        else:
            all_answer_boxes = Box(self.center_embedding.weight, as_point=True)
        membership_logit = self.cal_membership_logit(all_answer_boxes, all_query_boxes)
        transitive_relation_logit = self.cal_transitive_relation_logit(self.transitive_ids, self.inverse_ids)
        
        return positive_logit, negative_logit, membership_logit, transitive_relation_logit, subsampling_weight, all_idxs+all_union_idxs
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):
        model.train()
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(train_iterator)
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        for i, query in enumerate(batch_queries): # group queries with same structure
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        positive_logit, negative_logit, membership_logit, transitive_relation_logit, subsampling_weight, _ = model(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict, transitive=args.transitive)

        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        # membership_loss = 0.1 * membership_logit.mean()
        membership_loss = -F.logsigmoid(membership_logit).mean()
        relation_loss = -F.logsigmoid(transitive_relation_logit).mean()
        # relation_loss = transitive_relation_logit.mean()

        # lambda_reg = 0.1
        # reg = lambda_reg * (((model.center_mul.weight - 1.0) ** 2).mean() + ((model.center_add.weight) ** 2).mean() + ((model.offset_mul.weight - 1.0) ** 2).mean() + ((model.offset_add.weight) ** 2).mean())
        
        loss = (positive_sample_loss + negative_sample_loss)/2 + membership_loss  + relation_loss
        loss.backward()
        optimizer.step()
        
        
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'membership_loss': membership_loss.item(),
            'transitive_rel_loss': relation_loss.item(),
            'loss': loss.item()
        }
        
        return log

    @staticmethod
    def test_step(model, easy_answers, hard_answers, transitive_answers, args, test_dataloader, query_name_dict, save_result=False, save_str="", save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader, disable=not args.print_on_screen):
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()

                _, negative_logit,_, _, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict, transitive=args.transitive)
                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                if len(argsort) == args.test_batch_size: # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                    ranking = ranking.scatter_(1, argsort, model.batch_entity_range) # achieve the ranking of all entities
                else: # otherwise, create a new torch Tensor for batch_entity_range
                    if args.cuda:
                        ranking = ranking.scatter_(1, 
                                                   argsort, 
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 
                                                                                                      1).cuda()
                                                   ) # achieve the ranking of all entities
                    else:
                        ranking = ranking.scatter_(1, 
                                                   argsort, 
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 
                                                                                                      1)
                                                   ) # achieve the ranking of all entities
                for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
 
                    hard_answer = hard_answers[query]
                    easy_answer = easy_answers[query]
                    
                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)

                    if args.filter_deductive_triples:
                        transitive_answer = transitive_answers[query]
                        num_transitive = len(transitive_answer)
                        assert len(hard_answer.intersection(transitive_answer)) == 0
                        assert len(easy_answer.intersection(transitive_answer)) == 0

                    else:
                        transitive_answer = set()
                        num_transitive = 0
                        
                    assert len(hard_answer.intersection(easy_answer)) == 0
                    
                    cur_ranking = ranking[idx, list(easy_answer) + list(transitive_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy + num_transitive
                    if args.cuda:
                        answer_list = torch.arange(num_hard + num_easy + num_transitive).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_hard + num_easy + num_transitive).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1 # filtered setting
                    cur_ranking = cur_ranking[masks] # only take indices that belong to the hard answers

                    mrr = torch.mean(1./cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                        'num_hard_answer': num_hard,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]])/len(logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics
