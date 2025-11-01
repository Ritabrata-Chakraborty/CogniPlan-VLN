#!/usr/bin/env python3
"""
TensorBoard Training Visualizer - Creates Organized Subplot Figures

Usage:
1. Auto-discovery mode (no arguments): python visualize_training.py
   - Automatically finds all log directories and creates 3 organized subplot figures per log
   
2. Single file mode: python visualize_training.py <log_file_path>
   - Processes specific log file and creates 3 organized subplot figures

Output: Always creates 3 PNG files with subplots:
  - training_losses.png (all loss metrics in one figure)
  - performance_metrics.png (all performance metrics in one figure)
  - training_statistics.png (all other metrics in one figure)

Examples:
  python visualize_training.py
  python visualize_training.py checkpoints/cogniplan_exp_pred7_test/log/events.out.tfevents.1761177947.Mermista.289986.0
"""

import sys
import os
import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np


def load_tensorboard_data(log_file):
    """Load data from TensorBoard event file."""
    ea = EventAccumulator(log_file)
    ea.Reload()
    
    # Get all scalar tags
    scalar_tags = ea.Tags()['scalars']
    
    if not scalar_tags:
        return {}
    
    # Extract data for each tag
    data = {}
    for tag in scalar_tags:
        scalar_events = ea.Scalars(tag)
        if len(scalar_events) > 0:
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            data[tag] = {'steps': steps, 'values': values}
    
    return data


def create_losses_plot(data, save_dir):
    """Create a plot with loss-related metrics as subplots."""
    # Filter loss metrics
    loss_metrics = {k: v for k, v in data.items() if 'loss' in k.lower() or 'Loss' in k}
    
    if not loss_metrics:
        return
    
    # Create subplots
    n_metrics = len(loss_metrics)
    if n_metrics <= 2:
        rows, cols = 1, n_metrics
    elif n_metrics <= 4:
        rows, cols = 2, 2
    elif n_metrics <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    if n_metrics == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    fig.suptitle('Training Losses', fontsize=16, fontweight='bold')
    
    for idx, (metric, metric_data) in enumerate(loss_metrics.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        ax.plot(metric_data['steps'], metric_data['values'], linewidth=2, color='red', alpha=0.8)
        
        # Clean up metric name for title
        title = metric.replace('Losses/', '').replace('_', ' ')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(loss_metrics), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, 'training_losses.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_performance_plot(data, save_dir):
    """Create a plot with performance-related metrics as subplots."""
    # Filter performance metrics
    perf_metrics = {k: v for k, v in data.items() if 'perf' in k.lower() or 'Perf' in k or 'reward' in k.lower() or 'success' in k.lower() or 'accuracy' in k.lower()}
    
    if not perf_metrics:
        return
    
    # Create subplots
    n_metrics = len(perf_metrics)
    if n_metrics <= 2:
        rows, cols = 1, n_metrics
    elif n_metrics <= 4:
        rows, cols = 2, 2
    elif n_metrics <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    if n_metrics == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    fig.suptitle('Performance Metrics', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for idx, (metric, metric_data) in enumerate(perf_metrics.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        color = colors[idx % len(colors)]
        ax.plot(metric_data['steps'], metric_data['values'], linewidth=2, color=color, alpha=0.8)
        
        # Clean up metric name for title
        title = metric.replace('Perf/', '').replace('_', ' ')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(perf_metrics), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, 'performance_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_training_stats_plot(data, save_dir):
    """Create a plot with training statistics (entropy, gradients, etc.) as subplots."""
    # Filter training stats (everything else that's not loss or performance)
    exclude_keywords = ['loss', 'Loss', 'perf', 'Perf', 'reward', 'success', 'accuracy']
    training_stats = {}
    
    for k, v in data.items():
        if not any(keyword in k for keyword in exclude_keywords):
            training_stats[k] = v
    
    if not training_stats:
        return
    
    # Create subplots
    n_metrics = len(training_stats)
    if n_metrics <= 2:
        rows, cols = 1, n_metrics
    elif n_metrics <= 4:
        rows, cols = 2, 2
    elif n_metrics <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    if n_metrics == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    fig.suptitle('Training Statistics', fontsize=16, fontweight='bold')
    
    colors = ['darkgreen', 'darkblue', 'darkred', 'darkorange', 'darkviolet', 'darkcyan', 'darkgoldenrod', 'darkmagenta']
    
    for idx, (metric, metric_data) in enumerate(training_stats.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        color = colors[idx % len(colors)]
        ax.plot(metric_data['steps'], metric_data['values'], linewidth=2, color=color, alpha=0.8)
        
        # Clean up metric name for title
        title = metric.replace('Losses/', '').replace('_', ' ')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(training_stats), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, 'training_statistics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def process_single_log_file(log_file):
    """Process a single log file and create organized subplot figures."""
    print(f"Processing: {log_file}")
    
    try:
        # Load data
        data = load_tensorboard_data(log_file)
        
        if not data:
            print("No data found in log file!")
            return 0
        
        print(f"  Found {len(data)} metrics")
        
        # Get directory where log file is located
        save_dir = os.path.dirname(log_file)
        
        # Create the 3 organized subplot figures
        plots_created = 0
        
        create_losses_plot(data, save_dir)
        if os.path.exists(os.path.join(save_dir, 'training_losses.png')):
            plots_created += 1
            
        create_performance_plot(data, save_dir)
        if os.path.exists(os.path.join(save_dir, 'performance_metrics.png')):
            plots_created += 1
            
        create_training_stats_plot(data, save_dir)
        if os.path.exists(os.path.join(save_dir, 'training_statistics.png')):
            plots_created += 1
        
        return plots_created
        
    except Exception as e:
        print(f"  Error: {str(e)}")
        return 0
def find_log_directories():
    """Find all directories containing TensorBoard event files."""
    log_dirs = []
    
    # Common log locations
    search_paths = [
        "checkpoints/*/log",
        "sample_logs",
        "logs",
        "*/logs",
        "*/log"
    ]
    
    for pattern in search_paths:
        dirs = glob.glob(pattern)
        for dir_path in dirs:
            if os.path.isdir(dir_path):
                # Check if directory contains event files
                event_files = glob.glob(os.path.join(dir_path, "events.out.tfevents.*"))
                if event_files:
                    log_dirs.append(dir_path)
    
    return log_dirs


def process_log_directory(log_dir):
    """Process all event files in a log directory and create organized subplot figures."""
    # Find all event files in the directory
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    
    if not event_files:
        print(f"  No event files found in {log_dir}")
        return 0
    
    # Process the most recent event file (typically there's only one)
    event_file = max(event_files, key=os.path.getmtime)
    
    plots_created = process_single_log_file(event_file)
    
    return plots_created


def main():
    """Main function to handle both modes."""
    print("=== TensorBoard Training Visualizer ===")
    print("Creates organized subplot figures for training metrics\n")
    
    # Check if a specific log file is provided
    if len(sys.argv) == 2:
        # Single file mode
        log_file = sys.argv[1]
        
        if not os.path.exists(log_file):
            print(f"Error: Log file '{log_file}' does not exist!")
            sys.exit(1)
        
        print("Mode: Single file processing")
        plots_created = process_single_log_file(log_file)
        
        print(f"\n=== Complete ===")
        print(f"Created {plots_created} subplot figures")
        
    elif len(sys.argv) == 1:
        # Auto-discovery mode
        print("Mode: Auto-discovery")
        print("Searching for log directories...")
        
        log_dirs = find_log_directories()
        
        if not log_dirs:
            print("No log directories with event files found!")
            print("Searched in: checkpoints/*/log, sample_logs, logs, */logs, */log")
            return
        
        print(f"Found {len(log_dirs)} log director{'y' if len(log_dirs) == 1 else 'ies'}:")
        for log_dir in log_dirs:
            print(f"  - {log_dir}")
        
        print("\nProcessing logs...")
        
        total_plots = 0
        for log_dir in log_dirs:
            try:
                plots_created = process_log_directory(log_dir)
                total_plots += plots_created
                
                if plots_created > 0:
                    print(f"  ✓ Created {plots_created} subplot figure(s) in {log_dir}")
                
            except Exception as e:
                print(f"  ✗ Error processing {log_dir}: {str(e)}")
        
        print(f"\n=== Complete ===")
        print(f"Total subplot figures created: {total_plots}")
        print("Output: training_losses.png, performance_metrics.png, training_statistics.png")
    
    else:
        # Show usage
        print("Usage:")
        print("  Auto-discovery: python visualize_training.py")
        print("  Single file:    python visualize_training.py <log_file_path>")
        print()
        print("Examples:")
        print("  python visualize_training.py")
        print("  python visualize_training.py checkpoints/cogniplan_exp_pred7/log/events.out.tfevents.xxx.0")
        sys.exit(1)


if __name__ == "__main__":
    main()
