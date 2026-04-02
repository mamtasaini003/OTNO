"""
TOPOS Inference Script

Used for loading a pre-trained TOPOS model and running evaluations/inference
on the test set.

Usage examples:
  python topos_infer.py --dataset shapenet --topology spherical --checkpoint results/model_epoch_100.pt
"""

import argparse
import copy
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from timeit import default_timer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from topos.utils import LpLoss, count_model_params

# Import data loaders from the training script
from topos_train import (
    load_shapenet_data,
    load_flowbench_data,
    get_default_spherical_config,
    get_default_volumetric_config,
    resolve_routing_for_batch,
)
from topos.models.topos import TOPOS


def parse_args():
    parser = argparse.ArgumentParser(description="TOPOS Inference Script")

    # Core
    parser.add_argument('--dataset', type=str, required=True, choices=['shapenet', 'flowbench'],
                        help="Dataset to evaluate.")
    parser.add_argument('--topology', type=str, default='spherical',
                        choices=['spherical', 'toroidal', 'volumetric', 'auto'],
                        help="Topology routing mode.")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to the saved model checkpoint (.pt or .pth).")

    # Flowbench-specific args needed for correct dataloading
    parser.add_argument('--resolution', type=int, default=128, choices=[128, 256, 512],
                        help="Resolution (flowbench only).")
    parser.add_argument('--group_name', type=str, default='nurbs',
                        choices=['nurbs', 'harmonics', 'skelneton'],
                        help="Group name (flowbench only).")
    parser.add_argument('--latent_shape', type=str, default='square',
                        choices=['square', 'ring'],
                        help="Latent shape (flowbench only).")
    parser.add_argument('--expand_factor', type=int, default=2, choices=[1, 2, 3, 4],
                        help="Expand factor (flowbench only).")

    # Data locations (shared with training script)
    parser.add_argument(
        '--data_root',
        type=str,
        default=os.environ.get('TOPOS_DATA_ROOT'),
        help="Root directory containing dataset families (env: TOPOS_DATA_ROOT).",
    )
    parser.add_argument(
        '--shapenet_data_path',
        type=str,
        default=os.environ.get('TOPOS_SHAPENET_DATA'),
        help="Absolute/relative path to ShapeNet OT tensor file (env: TOPOS_SHAPENET_DATA).",
    )
    parser.add_argument(
        '--flowbench_root',
        type=str,
        default=os.environ.get('TOPOS_FLOWBENCH_ROOT'),
        help="Root of FlowBench LDC_NS_2D directory (env: TOPOS_FLOWBENCH_ROOT).",
    )
    
    # Save predictions
    parser.add_argument('--save_predictions', action='store_true',
                        help="If specified, saves inference predictions to npz format.")
    return parser.parse_args()


def infer_shapenet(args, device):
    print("Loading ShapeNet test data...")
    _, test_dataset, _, n_test, n_s_sqrt, pressure_encoder, _ = load_shapenet_data(args)

    # Initialize model
    spherical_config = get_default_spherical_config(in_channels=9, out_channels=1)
    volumetric_config = get_default_volumetric_config() if args.topology in ('auto', 'volumetric') else None

    model = TOPOS(
        spherical_config=spherical_config,
        toroidal_config=copy.deepcopy(spherical_config) if args.topology in ('auto', 'toroidal') else None,
        volumetric_config=volumetric_config,
        default_topology=args.topology,
    )
    
    print(f"Loading checkpoint: {args.checkpoint}")
    # Adjust error handling if you saved state_dict vs whole model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        # Fallback if bare state dict
        model.load_state_dict(checkpoint, strict=False)
        
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    pressure_encoder.to(device)
    data_loss = LpLoss(size_average=False)
    test_l2 = 0.0

    predictions_list = []
    ground_truth_list = []

    print("Running inference...")
    t1 = default_timer()
    
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            transports = batch_data['transports'].to(device)
            pressures = batch_data['pressures'].to(device)
            normals = batch_data['normals'][0].to(device)
            indices_encoder = batch_data['indices_encoder'][0].to(dtype=torch.long, device=device)
            indices_decoder = batch_data['indices_decoder'][0].to(dtype=torch.long, device=device)

            # Normal features
            normals = normals[indices_encoder]
            torus_normals = batch_data['nor'].reshape(-1, 3).to(device)
            normal_features = torch.cross(normals, torus_normals, dim=1).reshape(
                n_s_sqrt, n_s_sqrt, 3
            ).permute(2, 0, 1).unsqueeze(0)

            transports = torch.cat(
                (transports, batch_data['pos'].permute(0, 3, 1, 2).to(device), normal_features),
                dim=1,
            )

            topology, chi = resolve_routing_for_batch(
                args,
                batch_data,
                default_chi=2.0,
                default_topology="spherical",
            )
            
            # Run model
            out = model(transports, indices_decoder, topology=topology, chi=chi)
            
            # Decode output back to physical scale
            out_decoded = pressure_encoder.decode(out)
            
            # Loss computation
            loss = data_loss(out_decoded, pressures)
            test_l2 += loss.item()

            if args.save_predictions:
                predictions_list.append(out_decoded.cpu().numpy())
                ground_truth_list.append(pressures.cpu().numpy())
                
            if i % 20 == 0:
                print(f"Processed {i}/{len(test_loader)} samples...")

    time_taken = default_timer() - t1
    avg_l2 = test_l2 / n_test
    print(f"\n--- Inference Complete ---")
    print(f"Total time: {time_taken:.2f}s")
    print(f"Average Test normalized Lp-Loss: {avg_l2:.4f}")

    if args.save_predictions:
        os.makedirs("results", exist_ok=True)
        pred_path = f"results/shapenet_predictions_{args.topology}.npz"
        np.savez(pred_path, predictions=np.concatenate(predictions_list, axis=0), ground_truth=np.concatenate(ground_truth_list, axis=0))
        print(f"Predictions saved to {pred_path}")


def infer_flowbench(args, device):
    print("Loading FlowBench test data...")
    _, test_dataset, _, n_test, output_encoder, _ = load_flowbench_data(args)

    # Initialize model
    spherical_config = get_default_spherical_config(in_channels=7, out_channels=3)
    spherical_config['hidden_channels'] = 64  # Matches training
    volumetric_config = get_default_volumetric_config() if args.topology in ('auto', 'volumetric') else None

    model = TOPOS(
        spherical_config=spherical_config,
        toroidal_config=copy.deepcopy(spherical_config) if args.topology in ('auto', 'toroidal') else None,
        volumetric_config=volumetric_config,
        default_topology=args.topology,
    )
    
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
        
    model = model.to(device)
    model.eval()
    output_encoder.to(device)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    data_loss = LpLoss(d=3, size_average=False)
    test_l2 = 0.0
    
    predictions_list = []
    ground_truth_list = []

    print("Running inference...")
    t1 = default_timer()
    
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            inp = batch_data['inputs'].to(device)
            output = batch_data['outputs'][0].to(device)
            indices_decoder = batch_data['indices_decoder'][0].to(dtype=torch.long, device=device)

            topology, chi = resolve_routing_for_batch(
                args,
                batch_data,
                default_chi=0.0,
                default_topology="toroidal",
            )
            predict = model(
                inp.permute(0, 3, 1, 2).to(dtype=torch.float32, device=device),
                indices_decoder,
                topology=topology,
                chi=chi,
            )
            
            predict_decoded = output_encoder.decode(predict)
            
            loss = data_loss(output.unsqueeze(0), predict_decoded)
            test_l2 += loss.item()

            if args.save_predictions:
                predictions_list.append(predict_decoded.cpu().numpy())
                ground_truth_list.append(output.unsqueeze(0).cpu().numpy())
                
            if i % 20 == 0:
                print(f"Processed {i}/{len(test_loader)} samples...")

    time_taken = default_timer() - t1
    avg_l2 = test_l2 / n_test
    print(f"\n--- Inference Complete ---")
    print(f"Total time: {time_taken:.2f}s")
    print(f"Average Test normalized Lp-Loss: {avg_l2:.4f}")

    if args.save_predictions:
        os.makedirs("results", exist_ok=True)
        pred_path = f"results/flowbench_predictions_{args.topology}.npz"
        np.savez(pred_path, predictions=np.concatenate(predictions_list, axis=0), ground_truth=np.concatenate(ground_truth_list, axis=0))
        print(f"Predictions saved to {pred_path}")


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"TOPOS Inference — dataset={args.dataset}, topology={args.topology}")
    
    if args.dataset == 'shapenet':
        infer_shapenet(args, device)
    elif args.dataset == 'flowbench':
        infer_flowbench(args, device)
