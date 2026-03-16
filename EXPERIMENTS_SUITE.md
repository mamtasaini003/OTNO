# TOPOS: 4-Tier Validation Experimental Suite

To validate the TOPOS architecture, we must prove that its "topological awareness" and "optimal transport" components provide a measurable advantage over standard Graph Neural Networks (GNNs) and vanilla Fourier Neural Operators (FNOs). 

Here is our 4-tier experimental suite designed to stress-test the model's geometric and topological robustness:

## Tier 1: Topological Generalization (The Router Test)
The goal is to prove the Topological Router correctly identifies and handles different genus types in a single batch.
- **Dataset**: A mixed-genus collection including Spheres ($g=0$), Tori ($g=1$), and Double-Tori ($g=2$).
- **Task**: Solve the Poisson Equation ($\Delta u = f$) across all shapes.
- **Metric**: Compare Accuracy (RMSE) vs. a "Fixed-Branch" model (e.g., forcing a torus through a spherical FNO).
- **Success Criterion**: TOPOS should maintain < 5% error across all genera, whereas fixed-branch models should fail significantly on "mismatched" topologies.
- **Script**: `experiments/tier1_router_test/train_poisson_mixed_genus.py`

## Tier 2: Geometric Invariance (The OT Encoder Test)
This tests the OT Encoder's ability to "standardize" chaotic meshes into a clean latent space.
- **Dataset**: Use a single high-genus mesh (e.g., a complex mechanical part) but provide it in three resolutions: Coarse, Medium, and Ultra-Fine.
- **Task**: Predict Steady-State Heat Conduction.
- **Metric**: Resolution Invariance. Train on the Coarse mesh and test on the Ultra-Fine mesh without retraining.
- **Success Criterion**: TOPOS should exhibit "Zero-Shot Super-Resolution," outperforming GNNs (which are often tied to specific graph connectivity).
- **Script**: `experiments/tier2_geometric_invariance/train_heat_resolution_invariant.py`

## Tier 3: Physical Fidelity (The Latent Solver Test)
This tests the Spectral Neural Operator's ability to capture high-frequency physics in the latent domain.
- **Dataset**: 3D Navier-Stokes (Fluid Dynamics) on an irregular interior domain (e.g., blood flow through a branching artery).
- **Task**: Predict velocity and pressure fields over time.
- **Metric**: Spectral Energy Decay. Compare the predicted energy spectrum against a High-Fidelity DNS (Direct Numerical Simulation) ground truth.
- **Success Criterion**: The FNO stack should capture turbulent fluctuations more accurately than spatial-domain convolutions, especially in the latent grid.
- **Script**: `experiments/tier3_physical_fidelity/train_ns_irregular_domain.py`

## Tier 4: Comparative Benchmarking (The "SOTA" Test)
Directly compare TOPOS against its predecessors and competitors.

| Baseline       | What it lacks vs. TOPOS | Expected Outcome |
| :--- | :--- | :--- |
| **Vanilla FNO** | Cannot handle irregular/non-periodic boundaries. | TOPOS wins on boundary accuracy. |
| **GNO / GINO** | High computational cost on dense 3D meshes. | TOPOS (via FFT) should be $5\times$ to $10\times$ faster. |
| **OTNO** | Locked to Spherical ($g=0$) geometry. | TOPOS wins on any non-spherical domain. |

- **Script**: `experiments/tier4_benchmarks/run_comparative_baselines.py`

## Summary of Experimental Variables
- **Ablation Study**: Remove the Topological Router and use a "Global Average" solver to prove that genus-specific routing is necessary.
- **Noise Robustness**: Perturb the input mesh density $\mu$ with Gaussian noise to see if the OT Map $T$ remains stable.
