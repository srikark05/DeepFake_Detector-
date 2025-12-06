#!/usr/bin/env python3
"""
Create Key Results Diagram
Visualizes the key statistics from lines 76-90 of RESULTS_SECTION.md
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Key results data from lines 76-90
ACCURACY = 80.00
PRECISION_REAL = 90.48
RECALL_FAKE = 87.00
F1_SCORE = 82.61
FEATURE_COUNT = 7224
DATASET_SIZE = 200
TEST_SET_SIZE = 40
TRUE_POSITIVES = 19
TRUE_NEGATIVES = 13
FALSE_POSITIVES = 2
FALSE_NEGATIVES = 6

def create_key_results_diagram(output_path):
    """Create a comprehensive diagram showing all key results"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3, 
                          left=0.05, right=0.95, top=0.93, bottom=0.07)
    
    # 1. Main Performance Metrics (Large, Top Center)
    ax1 = fig.add_subplot(gs[0, 1])
    metrics = ['Accuracy', 'Precision\n(Real)', 'Recall\n(Fake)', 'F1-Score']
    values = [ACCURACY, PRECISION_REAL, RECALL_FAKE, F1_SCORE]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    bars = ax1.bar(metrics, values, color=colors, alpha=0.85, 
                   edgecolor='black', linewidth=2.5)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{value:.2f}%', ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
    
    ax1.set_ylabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Key Performance Metrics', fontsize=15, fontweight='bold', pad=15)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_axisbelow(True)
    
    # 2. Confusion Matrix (Top Left)
    ax2 = fig.add_subplot(gs[0, 0])
    cm = np.array([[TRUE_NEGATIVES, FALSE_POSITIVES],
                   [FALSE_NEGATIVES, TRUE_POSITIVES]])
    
    im = ax2.imshow(cm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=25, interpolation='nearest')
    
    # Add text with counts and percentages
    labels = [
        [f'{TRUE_NEGATIVES}\n(87%)', f'{FALSE_POSITIVES}\n(13%)'],
        [f'{FALSE_NEGATIVES}\n(24%)', f'{TRUE_POSITIVES}\n(76%)']
    ]
    
    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            percentage = labels[i][j].split('\n')[1]
            ax2.text(j, i, f'{int(value)}\n{percentage}',
                    ha="center", va="center", 
                    color="white" if value > 12 else "black",
                    fontsize=13, fontweight='bold')
    
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Predicted\nFake', 'Predicted\nReal'], 
                       fontsize=11, fontweight='bold')
    ax2.set_yticklabels(['Actual\nFake', 'Actual\nReal'], 
                       fontsize=11, fontweight='bold')
    ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=12)
    
    # Add labels
    label_text = ['TN', 'FP', 'FN', 'TP']
    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for (j, i), label in zip(positions, label_text):
        ax2.text(j, i-0.4, label, ha="center", va="center",
                fontsize=10, style='italic', color='darkblue', fontweight='bold')
    
    # 3. Classification Breakdown (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    categories = ['TP', 'TN', 'FP', 'FN']
    values_cm = [TRUE_POSITIVES, TRUE_NEGATIVES, FALSE_POSITIVES, FALSE_NEGATIVES]
    colors_cm = ['#2ecc71', '#27ae60', '#e74c3c', '#c0392b']
    
    bars = ax3.bar(categories, values_cm, color=colors_cm, alpha=0.85, 
                   edgecolor='black', linewidth=2)
    
    for bar, value in zip(bars, values_cm):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{int(value)}', ha='center', va='bottom', 
                fontsize=13, fontweight='bold')
    
    ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax3.set_title('Classification Breakdown', fontsize=13, fontweight='bold', pad=12)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.set_axisbelow(True)
    
    # Add full labels below
    full_labels = ['True\nPositive', 'True\nNegative', 'False\nPositive', 'False\nNegative']
    ax3.set_xticklabels([f'{cat}\n{label}' for cat, label in zip(categories, full_labels)],
                        fontsize=9)
    
    # 4. Quick Stats Table (Bottom Left)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    
    stats_data = [
        ['Metric', 'Value'],
        ['Accuracy', f'{ACCURACY:.2f}%'],
        ['Precision (Real)', f'{PRECISION_REAL:.2f}%'],
        ['Recall (Fake)', f'{RECALL_FAKE:.2f}%'],
        ['F1-Score', f'{F1_SCORE:.2f}%'],
        ['Feature Count', f'{FEATURE_COUNT:,}'],
        ['Dataset Size', f'{DATASET_SIZE} images'],
        ['Test Set', f'{TEST_SET_SIZE} images'],
    ]
    
    table = ax4.table(cellText=stats_data[1:], colLabels=stats_data[0],
                     cellLoc='left', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.08)
    
    # Style data rows
    for i in range(1, len(stats_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            table[(i, j)].set_height(0.06)
    
    ax4.set_title('Quick Reference Statistics', fontsize=13, fontweight='bold', pad=15)
    
    # 5. Detailed Breakdown (Bottom Middle)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    
    breakdown_data = [
        ['Classification', 'Count', 'Percentage'],
        ['True Positives', f'{TRUE_POSITIVES}', f'{TRUE_POSITIVES}/25 ({TRUE_POSITIVES/25*100:.0f}%)'],
        ['True Negatives', f'{TRUE_NEGATIVES}', f'{TRUE_NEGATIVES}/15 ({TRUE_NEGATIVES/15*100:.0f}%)'],
        ['False Positives', f'{FALSE_POSITIVES}', f'{FALSE_POSITIVES}/15 ({FALSE_POSITIVES/15*100:.0f}%)'],
        ['False Negatives', f'{FALSE_NEGATIVES}', f'{FALSE_NEGATIVES}/25 ({FALSE_NEGATIVES/25*100:.0f}%)'],
    ]
    
    table2 = ax5.table(cellText=breakdown_data[1:], colLabels=breakdown_data[0],
                      cellLoc='left', loc='center',
                      colWidths=[0.4, 0.2, 0.4])
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 2.5)
    
    # Style header
    for i in range(3):
        table2[(0, i)].set_facecolor('#e74c3c')
        table2[(0, i)].set_text_props(weight='bold', color='white')
        table2[(0, i)].set_height(0.08)
    
    # Color code rows
    row_colors = ['#d5f4e6', '#d5f4e6', '#ffe5e5', '#ffe5e5']
    for i in range(1, len(breakdown_data)):
        for j in range(3):
            table2[(i, j)].set_facecolor(row_colors[i-1])
            table2[(i, j)].set_height(0.07)
    
    ax5.set_title('Detailed Classification Breakdown', fontsize=13, fontweight='bold', pad=15)
    
    # 6. Feature Information (Bottom Right)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    feature_text = f"""
    Model Specifications
    
    • Classifier: SVM (RBF Kernel)
    • Features: {FEATURE_COUNT:,} per image
    • Hyperparameters:
      - C = 3.0
      - Gamma = "scale"
    
    Feature Sources:
    • 4 Filtration Types
    • 4 Feature Types per Filtration
    • 3 Homology Dimensions (H0, H1, H2)
    
    Dataset:
    • Total: {DATASET_SIZE} images
    • Real: 100 (50%)
    • Fake: 100 (50%)
    • Test Set: {TEST_SET_SIZE} images
    """
    
    ax6.text(0.05, 0.5, feature_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4, pad=10))
    ax6.set_title('Model & Dataset Info', fontsize=13, fontweight='bold', pad=15)
    
    # Main title
    fig.suptitle('TDA-Based DeepFake Detection - Key Results Summary', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved key results diagram: {output_path}")


def create_simple_results_diagram(output_path):
    """Create a simpler, cleaner diagram focusing on key metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Performance Metrics
    ax1 = axes[0]
    metrics = ['Accuracy', 'Precision\n(Real)', 'Recall\n(Fake)', 'F1-Score']
    values = [ACCURACY, PRECISION_REAL, RECALL_FAKE, F1_SCORE]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    bars = ax1.bar(metrics, values, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f'{value:.2f}%', ha='center', va='bottom', 
                fontsize=13, fontweight='bold')
    
    ax1.set_ylabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Performance Metrics', fontsize=15, fontweight='bold', pad=15)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_axisbelow(True)
    
    # Right: Confusion Matrix
    ax2 = axes[1]
    cm = np.array([[TRUE_NEGATIVES, FALSE_POSITIVES],
                   [FALSE_NEGATIVES, TRUE_POSITIVES]])
    
    im = ax2.imshow(cm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=25)
    
    # Add annotations
    annotations = [
        [f'{TRUE_NEGATIVES}\nTN\n(87%)', f'{FALSE_POSITIVES}\nFP\n(13%)'],
        [f'{FALSE_NEGATIVES}\nFN\n(24%)', f'{TRUE_POSITIVES}\nTP\n(76%)']
    ]
    
    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            text = annotations[i][j]
            ax2.text(j, i, text, ha="center", va="center",
                    color="white" if value > 12 else "black",
                    fontsize=12, fontweight='bold')
    
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Predicted Fake', 'Predicted Real'], 
                       fontsize=12, fontweight='bold')
    ax2.set_yticklabels(['Actual Fake', 'Actual Real'], 
                       fontsize=12, fontweight='bold')
    ax2.set_title('Confusion Matrix', fontsize=15, fontweight='bold', pad=15)
    
    # Add summary text
    summary_text = f'Accuracy: {ACCURACY:.2f}%\nTest Set: {TEST_SET_SIZE} images'
    ax2.text(1.15, 0.5, summary_text, transform=ax2.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('DeepFake Detection Model - Key Results', 
                fontsize=17, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved simple results diagram: {output_path}")


def main():
    """Generate key results diagrams"""
    output_dir = Path(__file__).parent / "Model_Results_Visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating key results diagrams...")
    print("=" * 60)
    
    create_key_results_diagram(output_dir / "Key_Results_Diagram.png")
    create_simple_results_diagram(output_dir / "Key_Results_Simple.png")
    
    print("=" * 60)
    print(f"Diagrams saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

