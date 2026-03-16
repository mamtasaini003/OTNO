import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_loss(csv_path, save_dir):
    """Parses training loss CSV and generates a training vs validation plot."""
    print(f"Reading loss data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Extract columns (Topos training produces: Epoch, Train Loss, Test Loss)
    epochs = df['Epoch']
    train_loss = df['Train Loss']
    test_loss = df['Test Loss']

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=2, alpha=0.9)
    plt.plot(epochs, test_loss, label='Test Loss', color='orange', linewidth=2, alpha=0.9)
    plt.yscale('log')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Log Scale)', fontsize=12)
    plt.title('TOPOS Training and Testing Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    
    out_path = os.path.join(save_dir, 'loss_plot.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Loss plot saved successfully to: {out_path}")
    plt.close()


def plot_visualizations(npz_path, save_dir, num_samples=3):
    """Parses inference prediction NPZ and generates ground truth, prediction, and error plots."""
    print(f"Reading prediction data from {npz_path}...")
    data = np.load(npz_path)
    predictions = data['predictions'] # shape [Batch, Channels, Nodes] depending on model
    ground_truth = data['ground_truth']
    
    # Calculate absolute error map
    error = np.abs(predictions - ground_truth)
    
    num_samples = min(num_samples, predictions.shape[0])
    
    for i in range(num_samples):
        pred_i = predictions[i]
        gt_i = ground_truth[i]
        err_i = error[i]
        
        # Squeeze out any unnecessary dimensions (like [1, N] -> [N])
        pred_i = np.squeeze(pred_i)
        gt_i = np.squeeze(gt_i)
        err_i = np.squeeze(err_i)
            
        if pred_i.ndim == 1:
            # Squeezed to 1D array - Plot as point series
            # E.g., ShapeNet pressures on surface 
            plt.figure(figsize=(18, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(gt_i, color='blue', alpha=0.8, marker='o', markersize=2, linestyle='None')
            plt.title('Ground Truth', fontsize=14)
            
            plt.subplot(1, 3, 2)
            plt.plot(pred_i, color='orange', alpha=0.8, marker='o', markersize=2, linestyle='None')
            plt.title('Prediction', fontsize=14)
            
            plt.subplot(1, 3, 3)
            plt.plot(err_i, color='red', alpha=0.8, marker='o', markersize=2, linestyle='None')
            plt.title('Absolute Error', fontsize=14)
            
        elif pred_i.ndim == 2:
            # Squeezed to 2D -> Determine if it's multiple channels of 1D coordinates [Channels, Nodes]
            # or a true 2D Image map [Width, Height]
            if pred_i.shape[0] <= 3: 
                # Flowbench data: [3, N] -> [U, V, P] channels
                fig, axes = plt.subplots(pred_i.shape[0], 3, figsize=(18, 4 * pred_i.shape[0]))
                if pred_i.shape[0] == 1:
                    axes_row = axes
                    axes_row[0].plot(gt_i[0], color='blue', marker='o', markersize=2, linestyle='None')
                    axes_row[0].set_title('Ground Truth (Channel 0)')
                    axes_row[1].plot(pred_i[0], color='orange', marker='o', markersize=2, linestyle='None')
                    axes_row[1].set_title('Prediction (Channel 0)')
                    axes_row[2].plot(err_i[0], color='red', marker='o', markersize=2, linestyle='None')
                    axes_row[2].set_title('Absolute Error (Channel 0)')
                else:
                    for c in range(pred_i.shape[0]):
                        axes[c, 0].plot(gt_i[c], color='blue', marker='o', markersize=2, linestyle='None')
                        axes[c, 0].set_title(f'Ground Truth (Channel {c})')
                        axes[c, 1].plot(pred_i[c], color='orange', marker='o', markersize=2, linestyle='None')
                        axes[c, 1].set_title(f'Prediction (Channel {c})')
                        axes[c, 2].plot(err_i[c], color='red', marker='o', markersize=2, linestyle='None')
                        axes[c, 2].set_title(f'Absolute Error (Channel {c})')
                plt.tight_layout()
            else: 
                # True 2D Image Map [Nx, Ny]
                plt.figure(figsize=(18, 5))
                
                plt.subplot(1, 3, 1)
                im = plt.imshow(gt_i, origin='lower', aspect='auto', cmap='viridis')
                plt.colorbar(im)
                plt.title('Ground Truth', fontsize=14)
                
                plt.subplot(1, 3, 2)
                im = plt.imshow(pred_i, origin='lower', aspect='auto', cmap='viridis')
                plt.colorbar(im)
                plt.title('Prediction', fontsize=14)
                
                plt.subplot(1, 3, 3)
                im = plt.imshow(err_i, origin='lower', aspect='auto', cmap='Reds')
                plt.colorbar(im)
                plt.title('Absolute Error', fontsize=14)
        
        else:
            print(f"Warning: High-dimensional visualization over 2D not explicitly handled for {pred_i.shape}. Customizing plot for raw tensor.")

        plt.suptitle(f'Sample {i+1} Visualization', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        out_path = os.path.join(save_dir, f'visualization_sample_{i+1}.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved successfully to: {out_path}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting script for TOPOS training metrics and test inferences.")
    parser.add_argument("--csv_file", type=str, default=None, help="Path to training CSV file with loss values.")
    parser.add_argument("--npz_file", type=str, default=None, help="Path to inference NPZ file with predictions.")
    parser.add_argument("--save_dir", type=str, default="results/plots", help="Directory to save the resulting plots.")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of specific test samples to visualize from the npz prediction dump.")
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.csv_file and os.path.exists(args.csv_file):
        plot_loss(args.csv_file, args.save_dir)
    elif args.csv_file:
        print(f"Error: CSV file '{args.csv_file}' not found.")
        
    if args.npz_file and os.path.exists(args.npz_file):
        plot_visualizations(args.npz_file, args.save_dir, args.num_samples)
    elif args.npz_file:
        print(f"Error: NPZ file '{args.npz_file}' not found.")
    
    if not args.csv_file and not args.npz_file:
        print("Please provide at least a --csv_file or an --npz_file to parse! Provide --help for usage details.")
