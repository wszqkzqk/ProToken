#!/usr/bin/env python3

import os
import glob
import argparse
import re
import sys
from pymol import cmd, finish_launching

def find_candidate_files(directory):
    """Finds pairs of protoken and smd cluster PDB files in a directory."""
    candidates = {}
    pro_pattern = os.path.join(directory, "candidate_*_protoken_frame_*.pdb")
    pro_files = glob.glob(pro_pattern)
    print(f"Found {len(pro_files)} potential protoken files matching pattern: {pro_pattern}")

    for pro_file in pro_files:
        basename = os.path.basename(pro_file)
        match = re.search(r"candidate_(\d+)_protoken_frame_(\d+)\.pdb", basename)
        if match:
            candidate_id = match.group(1)
            frame_num = match.group(2)
            smd_filename = f"candidate_{candidate_id}_smd_cluster_pro_{frame_num}.pdb"
            smd_file = os.path.join(directory, smd_filename)

            if os.path.exists(smd_file):
                if candidate_id not in candidates:
                    candidates[candidate_id] = {'pro': pro_file, 'smd': smd_file, 'frame': frame_num}
                    print(f"Found pair for Candidate {candidate_id} (Frame {frame_num}):")
                    print(f"  ProToken: {basename}")
                    print(f"  SMD Cluster: {smd_filename}")
                else:
                    # Handle potential duplicates if necessary, e.g., prefer lower frame number
                    # For now, just warn and keep the first one found
                    print(f"Warning: Duplicate candidate ID {candidate_id} found. Using frame {candidates[candidate_id]['frame']}.")
            else:
                print(f"Warning: SMD file '{smd_filename}' not found for ProToken file '{basename}'")
        else:
             print(f"Warning: Could not parse candidate ID and frame from '{basename}'")

    if not candidates:
        print(f"Error: No valid candidate pairs found in directory '{directory}'. Please check filenames.")

    return candidates

def visualize_candidate(candidate_id, pro_pdb, smd_pdb, output_dir, width=1200, height=900, dpi=300):
    """Generates a PyMOL visualization for a single candidate."""
    print(f"--- Processing Candidate {candidate_id} ---")
    cmd.reinitialize('everything') # Clear previous session

    pro_obj = f"pro_cand{candidate_id}"
    smd_cluster_obj = f"smd_cand{candidate_id}_cluster"
    smd_frame_prefix = f"smd_cand{candidate_id}_frame_"
    smd_group = f"smd_cand{candidate_id}_all_frames"

    # Load files
    print(f"Loading {os.path.basename(pro_pdb)} as {pro_obj}")
    cmd.load(pro_pdb, pro_obj)
    print(f"Loading {os.path.basename(smd_pdb)} as {smd_cluster_obj}")
    cmd.load(smd_pdb, smd_cluster_obj)

    # Set background color early (optional, can be done later too)
    cmd.bg_color("white")

    # Alignment (Align the whole SMD cluster object to ProToken before splitting)
    print(f"Aligning SMD cluster ({smd_cluster_obj}) to ProToken ({pro_obj}) using CA atoms...")
    align_mobile_sel = f"{smd_cluster_obj} and name CA"
    align_target_sel = f"{pro_obj} and name CA"
    try:
        # Align all states of the mobile object to the target object's first state
        cmd.align(align_mobile_sel, align_target_sel, cycles=5, object=f"{pro_obj}_and_{smd_cluster_obj}_aln")
        # Delete the alignment object created by cmd.align if not needed later
        cmd.delete(f"{pro_obj}_and_{smd_cluster_obj}_aln")
    except Exception as e:
        print(f"Warning: Alignment failed for candidate {candidate_id}. Skipping alignment. Error: {e}")

    # Process SMD cluster (Split states AFTER alignment)
    print("Splitting SMD cluster states...")
    cmd.split_states(smd_cluster_obj, prefix=smd_frame_prefix)
    cmd.delete(smd_cluster_obj) # Delete the original multi-state object
    smd_frames_selector = f"{smd_frame_prefix}*"
    cmd.group(smd_group, smd_frames_selector)
    print(f"Grouping SMD frames into '{smd_group}'")

    # Coloring and Representation
    print("Applying colors and representations...")
    cmd.color("green", pro_obj)
    cmd.show("cartoon", pro_obj)
    cmd.hide("lines", pro_obj) # Hide lines for clarity

    cmd.color("gray70", smd_group)
    cmd.show("cartoon", smd_group)
    cmd.hide("lines", smd_group)
    cmd.set("cartoon_transparency", 0.5, smd_group) # Make SMD frames transparent

    # Zoom and center
    print("Adjusting view...")
    cmd.orient(f"{pro_obj} or {smd_group}")
    cmd.zoom(f"{pro_obj} or {smd_group}")
    # Optional: Set a specific view for consistency if needed
    # view = cmd.get_view()
    # print(f"View for candidate {candidate_id}: {view}")
    # cmd.set_view(view)

    # Save image
    output_png = os.path.join(output_dir, f"candidate_{candidate_id}_visualization.png")
    print(f"Saving image to {output_png}...")
    cmd.png(output_png, width=width, height=height, dpi=dpi, ray=0)
    print(f"--- Finished Candidate {candidate_id} ---")


def main():
    parser = argparse.ArgumentParser(description="Automate PyMOL visualization for ProToken/SMD metastable candidates.")
    parser.add_argument("input_dir", help="Directory containing the extracted PDB files (e.g., 'extracted_metastable_frames').")
    parser.add_argument("-o", "--output_dir", help="Directory to save the output PNG images. If not specified, uses the input directory.")
    parser.add_argument("--width", type=int, default=2400, help="Width of the output image in pixels.")
    parser.add_argument("--height", type=int, default=1800, help="Height of the output image in pixels.")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution of the output image in DPI.")

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Set output directory to input directory if not specified
    output_dir = args.output_dir if args.output_dir else args.input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output images will be saved to: {output_dir}")

    # Launch PyMOL quietly in the background
    print("Launching PyMOL...")
    finish_launching(['pymol', '-qc']) # -q: quiet, -c: command line mode (no GUI)

    # Find candidate file pairs
    print(f"Scanning for candidate files in: {args.input_dir}")
    candidates = find_candidate_files(args.input_dir)

    if not candidates:
        print("No valid candidate file pairs found. Exiting.")
        cmd.quit()
        sys.exit(1)

    # Process each candidate
    for candidate_id, files in sorted(candidates.items(), key=lambda item: int(item[0])): # Sort by ID
        try:
            visualize_candidate(
                candidate_id,
                files['pro'],
                files['smd'],
                output_dir,
                width=args.width,
                height=args.height,
                dpi=args.dpi
            )
        except Exception as e:
            print(f"Error processing candidate {candidate_id}: {e}")
            # Optionally continue to the next candidate or stop
            # continue

    # Quit PyMOL
    print("Processing complete. Quitting PyMOL.")
    cmd.quit()

if __name__ == "__main__":
    main()
