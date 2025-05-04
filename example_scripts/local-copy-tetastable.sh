#!/bin/bash

# Source directory containing all the PDB pair directories
SRC_DIR="/tmp/lab/sponge_v1.4"

# Destination directory where we'll copy the metastable_cluster_analysis directories
DEST_DIR="/tmp/metastable-RAC47"  # Change this to your desired destination

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Find all PDB pair directories and process them
find "$SRC_DIR" -maxdepth 1 -type d -name "*_*-*_*" | while read -r pdb_pair_dir; do
    # Get just the PDB pair name (the directory name)
    pdb_pair_name=$(basename "$pdb_pair_dir")
    
    # Source metastable_cluster_analysis directory
    src_analysis_dir="$pdb_pair_dir/metastable_cluster_analysis"
    
    # Only proceed if the metastable_cluster_analysis directory exists
    if [ -d "$src_analysis_dir" ]; then
        # Destination directory path
        dest_pair_dir="$DEST_DIR/$pdb_pair_name"
        
        # Create destination directory
        mkdir -p "$dest_pair_dir"
        
        # Copy the metastable_cluster_analysis directory
        echo "Copying $src_analysis_dir to $dest_pair_dir/"
        cp -r "$src_analysis_dir" "$dest_pair_dir/"
    fi
done

echo "Copy operation completed. All metastable_cluster_analysis directories have been copied to $DEST_DIR"
