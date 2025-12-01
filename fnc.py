from itertools import combinations
import mdtraj as md
import numpy as np

def best_hummer_q(traj, top_file=None, native_file=None, frame_range=None):
    """
    Compute the Best Hummer Q from a trajectory and native structure.

    Parameters:
    - traj: str | md.Trajectory | np.ndarray
        Either a trajectory file path, an md.Trajectory, or a (n_frames, n_atoms, 3) position array
    - top_file: str
        Required if traj is a file path
    - native: str | md.Trajectory
        Native structure as a file path or md.Trajectory
    - native_file: str
        Used only if native is None
    - frame_range: tuple (start, end), optional
        Range of frames to evaluate
    - topology: md.Topology
        Required if traj is a position array

    Returns:
    - q: np.ndarray
        Q values for each frame
    """
    #print(traj,native_file,top_file)
    try:
            # Process trajectory input
        if isinstance(traj, str):
            if top_file is None:
                traj = md.load(traj)
            else:
                traj = md.load(traj, top=top_file)
        elif isinstance(traj, md.Trajectory):
            pass  # already in proper format
        elif isinstance(traj, np.ndarray):
            traj = md.Trajectory(xyz=traj, topology=md.load(top_file).topology)
        else:
            raise TypeError("Unsupported type for traj. Must be str, md.Trajectory, or np.ndarray.")


        native = md.load(native_file)
        #print("Native structure loaded from:", native_file)
        #print("Trajectory loaded from:", traj)
        # Constants
        BETA_CONST = 50  # 1/nm
        LAMBDA_CONST = 1.8
        NATIVE_CUTOFF = 0.45  # nm

        # Compute native contacts from heavy atoms
        heavy = native.topology.select_atom_indices('heavy')
        heavy_pairs = np.array([
            (i, j) for (i, j) in combinations(heavy, 2)
            if abs(native.topology.atom(i).residue.index - native.topology.atom(j).residue.index) > 3
        ])
        heavy_pairs_distances = md.compute_distances(native[0], heavy_pairs)[0]
        native_contacts = heavy_pairs[heavy_pairs_distances < NATIVE_CUTOFF]

        #print("Native contacts ", native_contacts)
        #print("Number of native contacts:", len(native_contacts))

        # Select heavy atoms
        heavy_traj = traj.topology.select_atom_indices('heavy')

        # Generate heavy atom pairs with residue index difference > 3
        heavy_pairs_traj = np.array([
            (i, j) for (i, j) in combinations(heavy_traj, 2)
            if abs(traj.topology.atom(i).residue.index - traj.topology.atom(j).residue.index) > 3
        ])

        # Compute distances ONLY for the last frame
        heavy_distances_traj = md.compute_distances(traj[-1], heavy_pairs_traj)

        # Identify contacts using cutoff
        contacts_traj = heavy_pairs_traj[heavy_distances_traj[0] < NATIVE_CUTOFF]

        # Convert to set of tuples for efficient intersection
        set1 = set(map(tuple, contacts_traj))
        set2 = set(map(tuple, native_contacts))

        # Find common rows
        common_rows = np.array(list(set1 & set2))

        novelty = len(common_rows) / len(native_contacts)


        # Slice frames if requested
        frames = traj[frame_range[0]:frame_range[1]] if frame_range is not None else traj

        # Calculate Q
        r = md.compute_distances(frames, native_contacts)
        r0 = md.compute_distances(native[0], native_contacts)
        q = np.mean(1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0))), axis=1)
        d_range = np.array([0,1])
        #print(q, d_range, novelty)
        return q, d_range

    except Exception as e:
        print(f"An error occurred during Hummer Q calculation: {e}")
        return None, None