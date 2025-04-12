#!/usr/bin/env python3

import sys
import os
import argparse

def convert_pdb_to_multimodel(input_pdb_path, output_pdb_path):
    """
    Convert a PDB file using 'END' as frame delimiter to a standard multi-model
    PDB file using 'MODEL' and 'ENDMDL' tags.

    Args:
        input_pdb_path (str): Path to the input PDB file.
        output_pdb_path (str): Path to the output PDB file.

    Returns:
        bool: True if conversion is successful, False otherwise.
    """
    print(f"Processing input file: {input_pdb_path}")
    print(f"Preparing to write output file: {output_pdb_path}")

    try:
        with open(input_pdb_path, 'r') as infile, open(output_pdb_path, 'w') as outfile:
            frame_number = 1
            # Flag to mark if we have already written the MODEL tag to start the current frame
            # Avoid writing empty MODEL/ENDMDL pairs at the beginning of the file (before the first atom line)
            # or between two END tags
            writing_frame_content = False

            for line in infile:
                # Remove whitespace characters (including newlines) from start and end of line
                stripped_line = line.strip()

                # Check if the line is the frame delimiter 'END'
                if stripped_line.startswith("END"):
                    # If we are currently writing a frame's content, end it with ENDMDL
                    if writing_frame_content:
                        outfile.write("ENDMDL\n")
                        # Reset the flag to indicate the current frame has ended
                        writing_frame_content = False
                        # print(f"  Completed frame {frame_number - 1}.") # Debug info
                # Check if the line is a valid PDB record (e.g., ATOM, HETATM, etc.)
                # Exclude empty lines and other non-atom record lines (unless they should be included)
                elif stripped_line and not stripped_line.startswith("MODEL") and not stripped_line.startswith("ENDMDL"):
                    # If we haven't started writing the current frame yet (i.e., this is the first valid line of a new frame)
                    if not writing_frame_content:
                        # Write MODEL tag to start a new frame
                        outfile.write(f"MODEL        {frame_number}\n")
                        # print(f"  Starting frame {frame_number}...") # Debug info
                        # Set the flag to indicate we are now writing frame content
                        writing_frame_content = True
                        # Increment frame number for the next frame
                        frame_number += 1
                    # Write the original PDB line to the output file (preserving the original format)
                    outfile.write(line)
                # else: # Skip empty lines or other unwanted lines
                    # pass

            # After the loop, check if the last frame has been closed with ENDMDL
            # This handles the case where the file doesn't end with 'END'
            if writing_frame_content:
                outfile.write("ENDMDL\n")
                # print(f"  Completed final frame {frame_number - 1} (end of file handling).") # Debug info

        print(f"\nConversion completed successfully!")
        print(f"Processed a total of {frame_number - 1} frames.")
        print(f"Output file saved to: {output_pdb_path}")
        return True

    except FileNotFoundError:
        print(f"\nError: Input file '{input_pdb_path}' not found. Please check the path and filename.")
        return False
    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")
        return False

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Object containing command line arguments
    """
    parser = argparse.ArgumentParser(
        description="PDB Frame Format Conversion Tool (END -> MODEL/ENDMDL)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input", 
        help="Path to the input PDB file"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Path to the output PDB file (default: input_traj.pdb)"
    )
    
    return parser.parse_args()

# --- Main Program Entry ---
if __name__ == "__main__":
    print("--- PDB Frame Format Conversion Tool (END -> MODEL/ENDMDL) ---")

    # Parse command line arguments
    args = parse_args()
    
    input_file = args.input
    
    # Set default output file if not specified
    if not args.output:
        output_file = input_file.replace('.pdb', '_traj.pdb')
        if output_file == input_file:  # Handle case where input filename doesn't have .pdb extension
            output_file = input_file + "_traj.pdb"
    else:
        output_file = args.output
    
    # Execute the conversion
    convert_pdb_to_multimodel(input_file, output_file)

    print("\n--- Program Ended ---")
