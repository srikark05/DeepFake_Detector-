#!/usr/bin/env python3
"""
Generate Visualizations for Model Results
Creates charts and images for key performance metrics and confusion matrix
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Results data
ACCURACY = 0.8000
PRECISION_REAL = 0.9048
RECALL_FAKE = 0.8700
F1_SCORE = 0.8261
FEATURE_COUNT = 7224
DATASET_SIZE = 200
TEST_SET_SIZE = 40
TRUE_POSITIVES = 19
TRUE_NEGATIVES = 13
FALSE_POSITIVES = 2
FALSE_NEGATIVES = 6

# Confusion matrix
CONFUSION_MATRIX = np.array([
    [TRUE_NEGATIVES, FALSE_POSITIVES],  # Actual Fake
    [FALSE_NEGATIVES, TRUE_POSITIVES]   # Actual Real
])

def create_performance_metrics_chart(output_path):
    """Create a bar chart of key performance metrics"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    metrics = ['Accuracy', 'Precision\n(Real)', 'Recall\n(Fake)', 'F1-Score']
    values = [ACCURACY * 100, PRECISION_REAL * 100, RECALL_FAKE * 100, F1_SCORE * 100]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{value:.2f}%',
               ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add horizontal line at 80%
    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved performance metrics chart: {output_path}")


def create_confusion_matrix_heatmap(output_path):
    """Create a heatmap visualization of the confusion matrix"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(CONFUSION_MATRIX, cmap='Blues', aspect='auto', vmin=0, vmax=25)
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            value = CONFUSION_MATRIX[i, j]
            percentage = (value / CONFUSION_MATRIX.sum(axis=1)[i]) * 100 if CONFUSION_MATRIX.sum(axis=1)[i] > 0 else 0
            text = ax.text(j, i, f'{int(value)}\n({percentage:.1f}%)',
                          ha="center", va="center", color="white" if value > 12 else "black",
                          fontsize=16, fontweight='bold')
    
    # Set labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Fake', 'Predicted Real'], fontsize=12, fontweight='bold')
    ax.set_yticklabels(['Actual Fake', 'Actual Real'], fontsize=12, fontweight='bold')
    
    # Add labels for each cell
    labels = [
        ['True Negative\n(Fake → Fake)', 'False Positive\n(Fake → Real)'],
        ['False Negative\n(Real → Fake)', 'True Positive\n(Real → Real)']
    ]
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i-0.35, labels[i][j], ha="center", va="center",
                   fontsize=10, style='italic', color='darkblue')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Number of Images', fontsize=12, fontweight='bold')
    
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    
    # Add summary statistics
    total = CONFUSION_MATRIX.sum()
    accuracy = (TRUE_POSITIVES + TRUE_NEGATIVES) / total * 100
    stats_text = f'Total Test Images: {int(total)}\nAccuracy: {accuracy:.2f}%'
    ax.text(1.5, 0.5, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix: {output_path}")


def create_per_class_metrics_chart(output_path):
    """Create a grouped bar chart comparing Fake vs Real class metrics"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Calculate metrics for Fake class
    fake_precision = TRUE_NEGATIVES / (TRUE_NEGATIVES + FALSE_NEGATIVES) if (TRUE_NEGATIVES + FALSE_NEGATIVES) > 0 else 0
    fake_recall = TRUE_NEGATIVES / (TRUE_NEGATIVES + FALSE_POSITIVES) if (TRUE_NEGATIVES + FALSE_POSITIVES) > 0 else 0
    fake_f1 = 2 * (fake_precision * fake_recall) / (fake_precision + fake_recall) if (fake_precision + fake_recall) > 0 else 0
    
    # Real class metrics (from results)
    real_precision = PRECISION_REAL
    real_recall = TRUE_POSITIVES / (TRUE_POSITIVES + FALSE_NEGATIVES) if (TRUE_POSITIVES + FALSE_NEGATIVES) > 0 else 0
    real_f1 = F1_SCORE
    
    categories = ['Precision', 'Recall', 'F1-Score']
    fake_values = [fake_precision * 100, fake_recall * 100, fake_f1 * 100]
    real_values = [real_precision * 100, real_recall * 100, real_f1 * 100]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, fake_values, width, label='Fake', 
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, real_values, width, label='Real', 
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved per-class metrics chart: {output_path}")


def create_dataset_summary_chart(output_path):
    """Create a pie chart and bar chart showing dataset composition"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart - Overall dataset
    labels = ['Real', 'Fake']
    sizes = [100, 100]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0.05)
    
    axes[0].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[0].set_title('Dataset Composition\n(Total: 200 images)', 
                     fontsize=14, fontweight='bold', pad=15)
    
    # Bar chart - Train/Test split
    categories = ['Training Set', 'Test Set']
    real_counts = [80, 25]  # Approximate from 80/20 split
    fake_counts = [80, 15]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = axes[1].bar(x - width/2, real_counts, width, label='Real', 
                       color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = axes[1].bar(x + width/2, fake_counts, width, label='Fake', 
                       color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    axes[1].set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    axes[1].set_title('Train/Test Split', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories, fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved dataset summary chart: {output_path}")


def create_comprehensive_results_dashboard(output_path):
    """Create a comprehensive dashboard with all key results"""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Performance Metrics (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Accuracy', 'Precision\n(Real)', 'Recall\n(Fake)', 'F1-Score']
    values = [ACCURACY * 100, PRECISION_REAL * 100, RECALL_FAKE * 100, F1_SCORE * 100]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = ax1.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Key Performance Metrics', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Confusion Matrix (Top Middle)
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(CONFUSION_MATRIX, cmap='Blues', aspect='auto', vmin=0, vmax=25)
    for i in range(2):
        for j in range(2):
            value = CONFUSION_MATRIX[i, j]
            ax2.text(j, i, f'{int(value)}', ha="center", va="center",
                    color="white" if value > 12 else "black", fontsize=14, fontweight='bold')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Predicted\nFake', 'Predicted\nReal'], fontsize=10, fontweight='bold')
    ax2.set_yticklabels(['Actual\nFake', 'Actual\nReal'], fontsize=10, fontweight='bold')
    ax2.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax2, fraction=0.046)
    
    # 3. Classification Breakdown (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    categories = ['TP', 'TN', 'FP', 'FN']
    values = [TRUE_POSITIVES, TRUE_NEGATIVES, FALSE_POSITIVES, FALSE_NEGATIVES]
    colors = ['#2ecc71', '#27ae60', '#e74c3c', '#c0392b']
    bars = ax3.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{int(value)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax3.set_title('Classification Breakdown', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    labels_full = ['True Positive\n(Real→Real)', 'True Negative\n(Fake→Fake)', 
                   'False Positive\n(Fake→Real)', 'False Negative\n(Real→Fake)']
    ax3.set_xticklabels([f'{cat}\n{label}' for cat, label in zip(categories, labels_full)], 
                        fontsize=9)
    
    # 4. Per-Class Metrics Comparison (Middle Left)
    ax4 = fig.add_subplot(gs[1, 0])
    fake_precision = TRUE_NEGATIVES / (TRUE_NEGATIVES + FALSE_NEGATIVES) if (TRUE_NEGATIVES + FALSE_NEGATIVES) > 0 else 0
    fake_recall = TRUE_NEGATIVES / (TRUE_NEGATIVES + FALSE_POSITIVES) if (TRUE_NEGATIVES + FALSE_POSITIVES) > 0 else 0
    fake_f1 = 2 * (fake_precision * fake_recall) / (fake_precision + fake_recall) if (fake_precision + fake_recall) > 0 else 0
    real_recall = TRUE_POSITIVES / (TRUE_POSITIVES + FALSE_NEGATIVES) if (TRUE_POSITIVES + FALSE_NEGATIVES) > 0 else 0
    
    categories = ['Precision', 'Recall', 'F1-Score']
    fake_values = [fake_precision * 100, fake_recall * 100, fake_f1 * 100]
    real_values = [PRECISION_REAL * 100, real_recall * 100, F1_SCORE * 100]
    
    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax4.bar(x - width/2, fake_values, width, label='Fake', color='#e74c3c', alpha=0.8)
    bars2 = ax4.bar(x + width/2, real_values, width, label='Real', color='#2ecc71', alpha=0.8)
    ax4.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Per-Class Metrics', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, fontsize=10)
    ax4.legend(fontsize=10)
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Dataset Summary (Middle)
    ax5 = fig.add_subplot(gs[1, 1])
    labels = ['Real', 'Fake']
    sizes = [100, 100]
    colors_pie = ['#2ecc71', '#e74c3c']
    ax5.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.0f%%',
           startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax5.set_title('Dataset Composition\n(200 images)', fontsize=12, fontweight='bold')
    
    # 6. Feature Information (Middle Right)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    feature_info = f"""
    Feature Extraction Summary
    
    Total Features: {FEATURE_COUNT:,} per image
    
    Feature Sources:
    • 4 Filtration Types
      - Height Filtration
      - Radial Filtration
      - Density Filtration
      - Dilation Filtration
    
    • 4 Feature Types per Filtration:
      - Persistence Landscapes
      - Persistence Entropy
      - Wasserstein Amplitudes
      - Betti Curves
    
    • Homology Dimensions: H0, H1, H2
    """
    ax6.text(0.1, 0.5, feature_info, transform=ax6.transAxes,
            fontsize=10, verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # 7. Key Statistics Table (Bottom)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Create table data
    table_data = [
        ['Metric', 'Value', 'Percentage/Count'],
        ['Overall Accuracy', f'{ACCURACY:.4f}', f'{ACCURACY*100:.2f}%'],
        ['Precision (Real)', f'{PRECISION_REAL:.4f}', f'{PRECISION_REAL*100:.2f}%'],
        ['Recall (Fake)', f'{RECALL_FAKE:.4f}', f'{RECALL_FAKE*100:.2f}%'],
        ['F1-Score', f'{F1_SCORE:.4f}', f'{F1_SCORE*100:.2f}%'],
        ['True Positives', f'{TRUE_POSITIVES}', f'{TRUE_POSITIVES}/25 ({TRUE_POSITIVES/25*100:.0f}%)'],
        ['True Negatives', f'{TRUE_NEGATIVES}', f'{TRUE_NEGATIVES}/15 ({TRUE_NEGATIVES/15*100:.0f}%)'],
        ['False Positives', f'{FALSE_POSITIVES}', f'{FALSE_POSITIVES}/15 ({FALSE_POSITIVES/15*100:.0f}%)'],
        ['False Negatives', f'{FALSE_NEGATIVES}', f'{FALSE_NEGATIVES}/25 ({FALSE_NEGATIVES/25*100:.0f}%)'],
        ['Feature Count', f'{FEATURE_COUNT:,}', 'per image'],
        ['Dataset Size', f'{DATASET_SIZE}', 'images (100 real, 100 fake)'],
        ['Test Set Size', f'{TEST_SET_SIZE}', 'images'],
    ]
    
    table = ax7.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.3, 0.25, 0.45])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header
    for i in range(3):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style alternating rows
    for i in range(1, len(table_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    ax7.set_title('Complete Results Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('TDA-Based DeepFake Detection Model - Results Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved comprehensive dashboard: {output_path}")


def main():
    """Generate all visualizations"""
    output_dir = Path(__file__).parent / "Model_Results_Visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating model results visualizations...")
    print("=" * 60)
    
    # Generate individual charts
    create_performance_metrics_chart(output_dir / "Performance_Metrics.png")
    create_confusion_matrix_heatmap(output_dir / "Confusion_Matrix.png")
    create_per_class_metrics_chart(output_dir / "Per_Class_Metrics.png")
    create_dataset_summary_chart(output_dir / "Dataset_Summary.png")
    create_comprehensive_results_dashboard(output_dir / "Results_Dashboard.png")
    
    print("=" * 60)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

