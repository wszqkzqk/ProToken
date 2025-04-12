#!/usr/bin/env python3

import os
import requests
import time
import sys
import gzip
import argparse
from pathlib import Path

def download_pdb(pdb_id, output_dir=".", max_retries=3, timeout=30):
    """
    Download PDB file from RCSB.org
    
    Args:
        pdb_id (str): PDB ID (4 characters)
        output_dir (str): Directory to save the PDB file
        max_retries (int): Maximum number of retries
        timeout (int): Timeout in seconds
    
    Returns:
        str: The path of the downloaded file, or None if the download fails
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Path to save the downloaded file
    output_path = os.path.join(output_dir, f"{pdb_id}.pdb")
    
    # Try multiple download sources
    sources = [
        f"https://files.rcsb.org/download/{pdb_id}.pdb",
        f"https://www.ebi.ac.uk/pdbe/entry-files/download/pdb{pdb_id.lower()}.ent",
        f"https://files.wwpdb.org/pub/pdb/data/structures/divided/pdb/{pdb_id[1:3].lower()}/pdb{pdb_id.lower()}.ent.gz"
    ]
    
    for source_url in sources:
        for retry in range(max_retries):
            try:
                print(f"Downloading {pdb_id} from {source_url}...")
                response = requests.get(source_url, timeout=timeout)
                response.raise_for_status()
                
                content = response.content
                
                # Check if the file is gzip compressed
                if source_url.endswith('.gz'):
                    content = gzip.decompress(content)
                
                # Save the file
                with open(output_path, "wb") as f:
                    f.write(content)
                
                print(f"Downloaded {pdb_id} to {output_path}")
                return output_path
            
            except requests.exceptions.RequestException as e:
                print(f"Attempt {retry+1}/{max_retries} failed: {e}")
                if retry == max_retries - 1:
                    print(f"Failed to download {pdb_id} from {source_url} after {max_retries} attempts.")
                else:
                    time.sleep(1)  # Wait before retrying
    
    print(f"Failed to download {pdb_id} from all sources")
    return None

def extract_chain(pdb_file_path, chain_id, output_dir="."):
    """
    Extract a specific chain from a PDB file
    
    Args:
        pdb_file_path (str): Path to the PDB file
        chain_id (str): Chain ID to extract
        output_dir (str): Directory to save the extracted chain
    
    Returns:
        str: Path of the extracted chain file, or None if an error occurs
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get PDB ID from file path
    pdb_id = os.path.basename(pdb_file_path).split('.')[0]
    
    # Path to save the extracted chain
    output_path = os.path.join(output_dir, f"{pdb_id}_{chain_id}.pdb")
    
    try:
        # Read the PDB file
        with open(pdb_file_path, "r") as f:
            pdb_content = f.readlines()
        
        # Extract the specified chain
        extracted_content = []
        
        # Handle special cases for chain ID
        chain_id_first_char = chain_id[0] if chain_id else ""
        
        for line in pdb_content:
            if line.startswith("ATOM"):
                # Use the split method to extract the chain ID instead of hardcoding positions
                # The ATOM line format in PDB files: ATOM  serial atom_name residue chain_id residue_seq ...
                parts = line.split()
                if len(parts) >= 5:  # Ensure there are enough fields
                    line_chain = parts[4]  # Chain ID can be multiple characters
                else:
                    # If split fails, fallback to the original method
                    line_chain = line[21:22].strip()
                
                # For chain IDs (can be multi-character in newer PDB formats)
                if line_chain == chain_id or (chain_id_first_char and line_chain.startswith(chain_id_first_char)):
                    extracted_content.append(line)
            elif line.startswith("END"):
                extracted_content.append(line)
        
        # Save the extracted chain
        with open(output_path, "w") as f:
            f.writelines(extracted_content)
        
        print(f"Extracted chain {chain_id} from {pdb_id} to {output_path}")
        
        # Warn about multi-character chain IDs
        if len(chain_id) > 1:
            print(f"Warning: {chain_id} is a multi-character chain ID; please verify {output_path}")
        
        return output_path
    
    except Exception as e:
        print(f"Error extracting chain {chain_id} from {pdb_id}: {e}")
        return None

def process_pdb_pair(pair, output_dir="."):
    """
    Process a pair of PDB IDs with chain specifications
    
    Args:
        pair (tuple): A tuple of PDB IDs (format: XXXX_Y)
        output_dir (str): Directory to save the processed file
    """
    results = []
    for entry in pair:
        # Split PDB ID and chain ID
        parts = entry.strip().split('_')
        if len(parts) < 2:
            print(f"Invalid entry format: {entry}")
            continue
        
        pdb_id = parts[0]
        chain_id = '_'.join(parts[1:])  # Handle multi-part cases
        
        # Download PDB file
        pdb_file_path = download_pdb(pdb_id, output_dir)
        if pdb_file_path:
            # Extract the specified chain
            result = extract_chain(pdb_file_path, chain_id, output_dir)
            results.append(result)
    
    return results

def parse_input_list(input_text):
    """
    Parse input text for PDB IDs with chain specifications
    
    Args:
        input_text (str): The input text containing PDB IDs with chain specifications
    
    Returns:
        list: A list of tuples (XXXX_Y, XXXX_Y)
    """
    # Split input text by lines
    lines = input_text.strip().split('\n')
    
    # Process each line
    pdb_pairs = []
    for line in lines:
        # Split line by semicolon
        entries = line.strip().split(';')
        if len(entries) == 2:
            pdb_pairs.append((entries[0], entries[1]))
        else:
            print(f"Invalid line format: {line}")
    
    return pdb_pairs

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Download PDB files and extract specific chains')
    parser.add_argument('--input', '-i', help='Input file containing PDB IDs with chain specifications')
    parser.add_argument('--output', '-o', default='downloaded_pdbs', help='Output directory for downloaded files')
    parser.add_argument('--retries', '-r', type=int, default=3, help='Maximum number of download retries')
    parser.add_argument('--timeout', '-t', type=int, default=30, help='Download timeout in seconds')
    
    args = parser.parse_args()
    
    # Check if input file is provided
    if args.input:
        try:
            with open(args.input, 'r') as f:
                input_text = f.read()
        except Exception as e:
            print(f"Error reading input file: {e}")
            return
    else:
        # Use hardcoded list
        input_text = """7Z3N_C;7Z3O_C
7R5J_F0;7R5K_F0
7YCO_B;8H6F_X
8TOC_b;8TV9_a
8TH8_S;8TID_s
7P37_A;7P3F_A
7UTI_Z;7UTL_d
8BDV_A;8BH7_A
7SJO_H;7SJP_H
7TUC_A;7TUE_A
8DKF_H;8DOW_C
8EFR_A;8EFT_A
8E2Y_A;8E31_A
8D01_L;8D0Y_L
8BFL_A;8BFP_A
7SJO_F;7SJP_L
8DUE_A;8DVF_A
7ZWM_E;7ZXF_E
7Y5A_D;7Y5B_D
8FWF_L;8FYM_L
8B6V_A;8B6W_A
8DKF_L;8DOW_D
7WKP_A;7WWU_I
7PIS_9;7PIT_9
8DKE_P;8DKI_P
8G4C_D;8G4D_D
7ZF5_D;7ZF6_H
8HKX_AS7P;8HKY_AS7P
8HFX_E;8IFY_E
8SQZ_C;8SRM_C
8HKX_S13P;8HKY_S13P
8HKY_AL1P;8HKZ_AL1P
8V2D_a;8V3B_A
8E39_A;8E3A_A
8TCA_L;8VEV_B
8EE5_L;8EF3_B
8EE5_H;8EF3_A
7ZF5_F;7ZF6_L
8HC3_E;8HC6_L
7QD7_AAA;7QH9_AAA
7T22_A;7T23_B
8UOP_A;8UOY_B
7S8G_L;7UDS_L
8I6O_B;8I6Q_B
8GAE_E;8GFT_E
8HC3_H;8HC5_H
8JSG_A;8JSH_A"""
    
    # Parse input list
    pdb_pairs = parse_input_list(input_text)
    
    # Process PDB pairs
    for i, pair in enumerate(pdb_pairs):
        print(f"Processing pair {i+1}/{len(pdb_pairs)}: {pair}")
        process_pdb_pair(pair, args.output)

if __name__ == "__main__":
    main()
