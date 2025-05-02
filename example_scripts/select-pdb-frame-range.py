import argparse
import sys
import os

def parse_frame_range(range_str):
    """Parse frame range string (e.g., '10' or '5-15') into (start, end) tuple (1-based)."""
    try:
        if '-' in range_str:
            start, end = map(int, range_str.split('-'))
            if start <= 0 or end <= 0 or start > end:
                raise ValueError("Range must be positive integers and start <= end.")
            return start, end
        else:
            end = int(range_str)
            if end <= 0:
                raise ValueError("Range must be positive integers.")
            return 1, end
    except ValueError as e:
        raise ValueError(f"Invalid frame range format '{range_str}': {e}. Use 'x' or 'x-y'.")

def read_pdb_frames(pdb_filepath):
    """Read PDB file and split into frames. Prefer MODEL/ENDMDL, otherwise use END."""
    if not os.path.exists(pdb_filepath):
        raise FileNotFoundError(f"Input file not found: {pdb_filepath}")

    with open(pdb_filepath, 'r') as f:
        content = f.read()

    frames = []
    # Try splitting based on MODEL/ENDMDL
    potential_frames_model = []
    current_frame_lines = []
    in_model = False
    lines = content.splitlines() # Split into lines for processing

    # Check if MODEL records exist
    has_model_records = any(line.strip().startswith("MODEL") for line in lines)

    if has_model_records:
        for line in lines:
            if line.strip().startswith("MODEL"):
                # If already in a model and we see a new MODEL, previous model missed ENDMDL
                # Or this is the first MODEL
                if in_model and current_frame_lines:
                     potential_frames_model.append("\n".join(current_frame_lines))
                elif not in_model and current_frame_lines: # Handle content before MODEL (usually ignored)
                    pass # Usually ignore content before MODEL as independent frames
                current_frame_lines = [] # Start new frame
                in_model = True
                # Do not add the MODEL line itself to frame content
            elif line.strip().startswith("ENDMDL"):
                if in_model:
                    potential_frames_model.append("\n".join(current_frame_lines))
                    current_frame_lines = []
                    in_model = False
                # else: Ignore ENDMDL not within MODEL
            elif in_model: # Only add lines between MODEL and ENDMDL
                current_frame_lines.append(line)
        # Handle the last model without ENDMDL (if exists)
        if in_model and current_frame_lines:
             potential_frames_model.append("\n".join(current_frame_lines))
        frames = potential_frames_model
        print(f"Detected {len(frames)} frames (based on MODEL/ENDMDL).")

    # If no MODEL/ENDMDL structure found, try splitting based on END
    if not frames:
        print("No MODEL/ENDMDL structure detected, trying to split frames based on END...")
        current_frame_lines = []
        for line in lines:
            # Use strip() to ensure only 'END' characters
            if line.strip() == "END":
                if current_frame_lines: # Avoid adding empty frames
                    frames.append("\n".join(current_frame_lines))
                current_frame_lines = [] # Reset for next frame
            else:
                # Add non-empty lines to current frame
                if line.strip():
                    current_frame_lines.append(line)

        # Add the last frame split by END (if file ends without END)
        if current_frame_lines:
            frames.append("\n".join(current_frame_lines))
        print(f"Detected {len(frames)} frames (based on END).")

    if not frames:
         print("Warning: No frames could be split from the input file.", file=sys.stderr)

    return frames

def write_selected_frames(frames, start_frame, end_frame, output_filepath):
    """Write selected frames to a new PDB file using MODEL/ENDMDL format."""
    total_frames = len(frames)
    if start_frame > total_frames:
        print(f"Warning: start frame ({start_frame}) is greater than total frames ({total_frames}). No frames written.", file=sys.stderr)
        # Create an empty file or handle as needed
        with open(output_filepath, 'w') as outfile:
            pass # Write empty file
        return

    # Adjust end frame to not exceed total frames
    actual_end_frame = min(end_frame, total_frames)

    print(f"Writing frames {start_frame} to {actual_end_frame} into {output_filepath}...")
    output_frame_index = 1 # MODEL index in output file starts from 1
    with open(output_filepath, 'w') as outfile:
        # PDB frame index is 1-based, list index is 0-based
        for i in range(start_frame - 1, actual_end_frame):
            outfile.write(f"MODEL        {output_frame_index}\n")
            # Write frame content, ensuring it ends with a newline
            frame_content = frames[i].strip() # Remove possible leading/trailing whitespace
            outfile.write(frame_content)
            outfile.write("\n") # Ensure newline after frame content
            outfile.write("ENDMDL\n")
            output_frame_index += 1
    print(f"Successfully wrote {output_frame_index - 1} frames.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract a specified range of frames from a multi-frame PDB file."
    )
    parser.add_argument("input_pdb",    help="Input multi-frame PDB file path.")
    parser.add_argument("frame_range",  help="Frame range to extract. Format: 'x' (1 to x) or 'x-y' (x to y).")
    parser.add_argument("output_pdb",   help="Output PDB file path.")

    args = parser.parse_args()

    try:
        start_frame, end_frame = parse_frame_range(args.frame_range)
        print(f"Requested frame range: {start_frame}-{end_frame}")

        frames = read_pdb_frames(args.input_pdb)

        if not frames:
            print("No frames were extracted from the input file. Exiting.", file=sys.stderr)
            sys.exit(1)

        total_frames_found = len(frames)
        print(f"Read {total_frames_found} frames from '{args.input_pdb}'.")

        if start_frame > total_frames_found:
            print(f"Error: requested start frame ({start_frame}) exceeds total frames found ({total_frames_found}).", file=sys.stderr)
            sys.exit(1)

        write_selected_frames(frames, start_frame, end_frame, args.output_pdb)

    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
