import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_grid(nodes_file="Nodes.txt", elements_file="Elements.txt"):
    # --- 1. Load Nodes ---
    nodes = {}
    with open(nodes_file, 'r') as f:
        lines = f.readlines()
        # Skip header (count)
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 4:
                # Format: ID x y z
                nid = int(parts[0])
                nodes[nid] = [float(parts[1]), float(parts[2]), float(parts[3])]

    # --- 2. Load Elements ---
    triangles = []
    with open(elements_file, 'r') as f:
        lines = f.readlines()
        # Skip header (count)
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 4:
                # Format: ID n1 n2 n3 ...
                triangles.append([int(parts[1]), int(parts[2]), int(parts[3])])

    # --- 3. Prepare Data for Plotting ---
    # Map high IDs (e.g., 200001) to 0-based indices for matplotlib
    node_ids = sorted(nodes.keys())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    
    coords = np.array([nodes[nid] for nid in node_ids])
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    
    # Map triangle node IDs to indices
    tri_indices = []
    for t in triangles:
        try:
            tri_indices.append([id_to_idx[t[0]], id_to_idx[t[1]], id_to_idx[t[2]]])
        except KeyError:
            continue # Skip if node missing

    # --- 4. Plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    ax.plot_trisurf(x, y, z, triangles=tri_indices, cmap='viridis', 
                    edgecolor='k', linewidth=0.2, alpha=0.8)
    
    # Make axes equal aspect ratio (looks like a sphere, not an oval)
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_title(f"Evaluation Grid (Nodes: {len(node_ids)})")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    
    plt.show()

if __name__ == "__main__":
    visualize_grid()