import glob
import os
import trimesh
import sys

sys.path.append('/home/mamta/work/OTNO')
from topos.router.topology_check import TopologicalRouter, compute_euler_characteristic

mesh_files = sorted(glob.glob('/media/HDD/mamta_backup/datasets/otno/car-pressure-data/data/mesh_*.ply'))

router = TopologicalRouter()
topologies = {}

print(f"Testing {len(mesh_files)} meshes...")
for f in mesh_files[:100]:
    try:
        mesh = trimesh.load(f, force='mesh', process=False)
        chi = compute_euler_characteristic(mesh)
        route = router.route(chi=chi)
        topologies[route] = topologies.get(route, 0) + 1
        print(f"{os.path.basename(f)}: chi={chi}, route={route}")
    except Exception as e:
        print(f"Error processing {os.path.basename(f)}: {e}")

print("Summary of routing:")
print(topologies)
