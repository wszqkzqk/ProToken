import subprocess

# List of systems
systems = [
    "7P37_A-7P3F_A",
    "7QD7_AAA-7QH9_AAA",
    "7S8G_L-7UDS_L",
    "7SJO_H-7SJP_H",
    "7UTI_Z-7UTL_d",
    "7ZF5_D-7ZF6_H",
    "7ZF5_F-7ZF6_L",
    "7ZWM_E-7ZXF_E",
    "8D01_L-8D0Y_L",
    "8DKF_H-8DOW_C",
    "8DKF_L-8DOW_D",
    "8FWF_L-8FYM_L",
    "8HFX_E-8IFY_E",
    "8HKX_S13P-8HKY_S13P",
    "8TCA_L-8VEV_B",
    "8V2D_a-8V3B_A"
]

# Iterate through each system and execute commands
for system in systems:
    # Build command as an array
    # command = [
    #     "python",
    #     "/lustre/grp/gyqlab/chenzhenyu/zhouqk/ProToken/example_scripts/pca-dtw.py",
    #     f"/lustre/grp/gyqlab/chenzhenyu/zhouqk/sponge_v1.4/{system}/start.pdb",
    #     f"/lustre/grp/gyqlab/chenzhenyu/zhouqk/sponge_v1.4/{system}/end.pdb",
    #     f"/lustre/grp/gyqlab/chenzhenyu/zhouqk/sponge_v1.4/{system}/{system}.pdb",
    #     f"/lustre/grp/gyqlab/chenzhenyu/zhouqk/RAC-47-Results/{system}/result.pdb",
    #     "-o",
    #     f"/lustre/grp/gyqlab/chenzhenyu/zhouqk/sponge_v1.4/{system}/dtw_pca_dist_analysis",
    # ]
    # command = [
    #     "python",
    #     "/lustre/grp/gyqlab/chenzhenyu/zhouqk/ProToken/example_scripts/SMD-ProToken-in-RMSD-space.py",
    #     f"/lustre/grp/gyqlab/chenzhenyu/zhouqk/sponge_v1.4/{system}/start.pdb",
    #     f"/lustre/grp/gyqlab/chenzhenyu/zhouqk/sponge_v1.4/{system}/end.pdb",
    #     f"/lustre/grp/gyqlab/chenzhenyu/zhouqk/sponge_v1.4/{system}/{system}.pdb",
    #     f"/lustre/grp/gyqlab/chenzhenyu/zhouqk/RAC-47-Results/{system}/result.pdb",
    # ]
    command = [
        "python",
        "/lustre/grp/gyqlab/chenzhenyu/zhouqk/ProToken/example_scripts/find_metastable.py",
        f"/lustre/grp/gyqlab/chenzhenyu/zhouqk/sponge_v1.4/{system}/start.pdb",
        f"/lustre/grp/gyqlab/chenzhenyu/zhouqk/sponge_v1.4/{system}/end.pdb",
        f"/lustre/grp/gyqlab/chenzhenyu/zhouqk/sponge_v1.4/{system}/{system}.pdb",
        f"/lustre/grp/gyqlab/chenzhenyu/zhouqk/RAC-47-Results/{system}/result.pdb",
        f"/lustre/grp/gyqlab/chenzhenyu/zhouqk/sponge_v1.4/{system}/dtw_pca_dist_analysis/selected_ca_pairs_indices.pkl",
        f"/lustre/grp/gyqlab/chenzhenyu/zhouqk/sponge_v1.4/{system}/dtw_pca_dist_analysis/pca_model.pkl",
        "-o",
        f"/lustre/grp/gyqlab/chenzhenyu/zhouqk/sponge_v1.4/{system}/metastable_cluster_analysis",
    ]
    # Output the current system being processed
    print(f"Processing system: {system}")
    try:
        # Execute command
        subprocess.run(command, check=True)
        print(f"Successfully processed system: {system}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing system {system}: {e}")

print("All systems processed successfully")
