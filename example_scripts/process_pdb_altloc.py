#!/usr/bin/env python3

import argparse
import sys
import os

def format_atom_line(record_name, atom_serial, atom_name, res_name, chain_id,
                     res_seq, i_code, x_coord, y_coord, z_coord, b_factor,
                     element_symbol, charge, seg_id="    "):
    """
    Formats an ATOM/HETATM line according to PDB specification,
    setting altLoc=' ' and occupancy=1.00.

    Args:
        record_name (str): "ATOM  " or "HETATM"
        atom_serial (int): Atom serial number
        atom_name (str): Atom name (exactly 4 chars, padded if needed)
        res_name (str): Residue name (exactly 3 chars)
        chain_id (str): Chain ID (1 char)
        res_seq (int): Residue sequence number
        i_code (str): Insertion code (1 char, usually space)
        x_coord (float): X coordinate
        y_coord (float): Y coordinate
        z_coord (float): Z coordinate
        b_factor (float): Temperature factor
        element_symbol (str): Element symbol (exactly 2 chars, right justified)
        charge (str): Atom charge (exactly 2 chars, e.g., "2+" or "  ")
        seg_id (str): Segment ID (exactly 4 chars, optional, defaults to spaces)

    Returns:
        str: Formatted PDB line (without newline).
    """
    # Ensure atom_name has correct padding (often space before for short names)
    # PDB standard: Atom names start with element symbol left-justified if < 4 chars
    # But many programs use fixed padding like " CA ". Best to use original if possible,
    # but here we reformat. Centering is often used for 2/3 char names.
    # Let's try standard alignment: Left-justified for element if < 4 chars.
    # Example: " CA ", " OXT", " N  "
    # Let's re-evaluate based on common PDB usage: 4 chars, often padded with spaces.
    # " CA " is typical for C-alpha. Let's assume the input `atom_name` arg is already padded.

    line = (
        f"{record_name:<6}"        # 1-6   Record name
        f"{atom_serial:>5} "       # 7-11  Atom serial number, 12 Space
        f"{atom_name:<4}"         # 13-16 Atom name (use original padding if passed)
        f"{' ':<1}"               # 17    Alternate location indicator (forced to space)
        f"{res_name:>3} "         # 18-20 Residue name, 21 Space
        f"{chain_id:<1}"          # 22    Chain identifier
        f"{res_seq:>4}"           # 23-26 Residue sequence number
        f"{i_code:<1}   "         # 27    Insertion code, 28-30 Spaces
        f"{x_coord:>8.3f}"        # 31-38 X coordinate
        f"{y_coord:>8.3f}"        # 39-46 Y coordinate
        f"{z_coord:>8.3f}"        # 47-54 Z coordinate
        f"{1.00:>6.2f}"           # 55-60 Occupancy (forced to 1.00)
        f"{b_factor:>6.2f}      " # 61-66 Temperature factor, 67-72 Spaces
        f"{seg_id:<4}"            # 73-76 Segment identifier
        f"{element_symbol:>2}"    # 77-78 Element symbol (right justified)
        f"{charge:>2}"            # 79-80 Charge (right justified)
    )
    # Ensure the line doesn't exceed 80 characters (it should be exactly 80 if formatting is correct)
    return line[:80]


def process_pdb(input_pdb_path, output_pdb_path):
    """
    Reads a PDB file, processes alternate conformations, and writes a new PDB file.

    Keeps only one conformation per atom:
    1. Prioritizes conformation 'A' if present.
    2. Otherwise, keeps the first conformation encountered for that atom.
    Sets altLoc to ' ' and occupancy to 1.00 for all kept ATOM/HETATM records.
    Preserves formatting strictly.
    """
    if not os.path.exists(input_pdb_path):
        print(f"Error: Input file not found: {input_pdb_path}")
        sys.exit(1)

    # Set to store unique identifiers for atoms that had alternate locations
    # and for which we have already selected a conformation.
    # Key: (Chain ID, Residue Sequence Number, Insertion Code, Atom Name)
    seen_alt_atoms = set()

    print(f"Processing PDB file: {input_pdb_path}")

    try:
        with open(input_pdb_path, 'r') as infile, open(output_pdb_path, 'w') as outfile:
            for line in infile:
                # Process only ATOM and HETATM records for conformations
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    try:
                        record_name = line[0:6]
                        atom_serial = int(line[6:11].strip())
                        atom_name   = line[12:16] # Keep original padding
                        alt_loc     = line[16:17]
                        res_name    = line[17:20].strip() # Use stripped for logic, formatted later
                        chain_id    = line[21:22]
                        res_seq     = int(line[22:26].strip())
                        i_code      = line[26:27]
                        x_coord     = float(line[30:38].strip())
                        y_coord     = float(line[38:46].strip())
                        z_coord     = float(line[46:54].strip())
                        # Occupancy is not needed for logic, B-factor needed for output
                        b_factor    = float(line[60:66].strip())
                        # Optional fields - keep original if present
                        seg_id         = line[72:76] if len(line) >= 76 else "    "
                        element_symbol = line[76:78] if len(line) >= 78 else "  "
                        charge         = line[78:80] if len(line) >= 80 else "  "

                        # Ensure required fields have expected lengths for formatting later
                        if len(res_name) > 3: res_name = res_name[:3] # Should not happen in valid PDB

                        # Unique identifier for this specific atom instance
                        atom_key = (chain_id, res_seq, i_code, atom_name)

                        process_this_line = False

                        if alt_loc == ' ' or alt_loc == 'A':
                            # If it's the standard conformation or explicit 'A',
                            # process it unless we've already processed an 'A' or
                            # the first encountered alt loc for this atom.
                            if atom_key not in seen_alt_atoms:
                                process_this_line = True
                                # If it has an altLoc ('A'), mark this atom as seen
                                # to prevent processing 'B', 'C', etc. later.
                                # If alt_loc was ' ', it didn't have alternates anyway.
                                if alt_loc != ' ':
                                     seen_alt_atoms.add(atom_key)
                        elif alt_loc.strip() and alt_loc != 'A':
                            # It's an alternate location indicator other than 'A' (e.g., 'B', 'C')
                            if atom_key not in seen_alt_atoms:
                                # This is the first conformation we encounter for this atom
                                # (meaning 'A' wasn't present or processed yet). Keep it.
                                process_this_line = True
                                seen_alt_atoms.add(atom_key)
                            # else: This is a subsequent conformation ('B', 'C', etc.)
                            # and we've already chosen one ('A' or the first one), so skip it.

                        if process_this_line:
                            # Format the line according to PDB spec with modifications
                            output_line = format_atom_line(
                                record_name=record_name,
                                atom_serial=atom_serial,
                                atom_name=atom_name, # Use original padding
                                res_name=res_name, # Use stripped name
                                chain_id=chain_id,
                                res_seq=res_seq,
                                i_code=i_code,
                                x_coord=x_coord,
                                y_coord=y_coord,
                                z_coord=z_coord,
                                b_factor=b_factor,
                                element_symbol=element_symbol.strip().rjust(2), # Ensure right justify
                                charge=charge.strip().rjust(2), # Ensure right justify
                                seg_id=seg_id # Use original seg_id
                            )
                            outfile.write(output_line + '\n')
                        # If not process_this_line, we simply skip writing it.

                    except (ValueError, IndexError) as e:
                        print(f"Warning: Skipping malformed ATOM/HETATM line: {line.strip()}", file=sys.stderr)
                        print(f"         Error: {e}", file=sys.stderr)
                        # Optionally write the original malformed line? Best to skip.
                        # outfile.write(line) # Or skip

                else:
                    # For any other line type (HEADER, REMARK, TER, END, etc.),
                    # write it directly to the output file.
                    # Use rstrip('\n') + '\n' to ensure consistent line endings?
                    # No, PDB files can have specific whitespace, just write the line.
                    outfile.write(line)

        print(f"Successfully processed PDB file. Output saved to: {output_pdb_path}")

    except IOError as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Process a PDB file to remove alternate conformations, keeping only one "
                    "(prioritizing 'A', then first encountered) per atom. Ensures strict PDB format.",
        formatter_class=argparse.RawTextHelpFormatter
        )
    parser.add_argument("input_pdb", help="Path to the input PDB file.")
    parser.add_argument("output_pdb", help="Path for the processed output PDB file.")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    process_pdb(args.input_pdb, args.output_pdb)

if __name__ == "__main__":
    main()
