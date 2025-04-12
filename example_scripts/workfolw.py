#!/usr/bin/env python3

import os
import subprocess
import shutil
import logging
from datetime import datetime

# Set up logging
log_dir = "/lustre/grp/gyqlab/chenzhenyu/zhouqk/ProToken/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# List of PDB pairs
pdb_pairs = [
    "7SJO_F;7SJP_L",
    "7SJO_H;7SJP_H",
    "7TUC_A;7TUE_A",
    "7WKP_A;7WWU_I",
    "7YCO_B;8H6F_X",
    "7ZF5_F;7ZF6_L",
    "8BDV_A;8BH7_A",
    "8DKF_L;8DOW_D",
    "8EE5_H;8EF3_A",
    "8EE5_L;8EF3_B",
    "8FWF_L;8FYM_L",
    "8HC3_H;8HC5_H",
    "8JSG_A;8JSH_A",
    "8TCA_L;8VEV_B",
]

# Path definitions
BASE_PATH = "/lustre/grp/gyqlab/chenzhenyu/zhouqk/ProToken"
PKL_PATH = f"{BASE_PATH}/pkls/RAC-47/generator_inputs"
PDBS_RESULT_PATH = f"{BASE_PATH}/pdbs/result"
FINAL_RESULTS_PATH = f"{BASE_PATH}/results/RAC-47"

# Ensure that result directories exist
os.makedirs(PDBS_RESULT_PATH, exist_ok=True)
os.makedirs(FINAL_RESULTS_PATH, exist_ok=True)

# Record successful and failed systems
successful_systems = []
failed_systems = []

def run_command(cmd, error_msg):
    """Run command and handle potential errors"""
    try:
        logging.info(f"Running command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError:
        logging.error(error_msg)
        return False

# Main workflow process
for pair in pdb_pairs:
    try:
        # Parse PDB pair
        pdb1, pdb2 = pair.split(';')
        pair_name = f"{pdb1}-{pdb2}"
        logging.info(f"Processing PDB pair: {pair_name}")
        
        # Construct pkl paths
        pkl1_path = f"{PKL_PATH}/{pdb1}.pkl"
        pkl2_path = f"{PKL_PATH}/{pdb2}.pkl"
        
        # Check if pkl files exist
        if not os.path.exists(pkl1_path) or not os.path.exists(pkl2_path):
            logging.error(f"Cannot find pkl file: {pkl1_path} or {pkl2_path}")
            failed_systems.append(pair_name)
            continue
        
        # 1. Run generate-meta-stable.py script
        cmd1 = f"python '{BASE_PATH}/example_scripts/generate-meta-stable.py' --pkl1 {pkl1_path} --pkl2 {pkl2_path} -o {PDBS_RESULT_PATH}"
        if not run_command(cmd1, f"Failed to run generate-meta-stable.py: {pair_name}"):
            failed_systems.append(pair_name)
            continue
        
        # 2. Run decode_structure_sbatch.sh script
        cmd2 = f"{BASE_PATH}/example_scripts/decode_structure_sbatch.sh"
        if not run_command(cmd2, f"Failed to run decode_structure_sbatch.sh: {pair_name}"):
            failed_systems.append(pair_name)
            continue
        
        # 3. Run merge-pdbs.py script
        output_pdb = f"{PDBS_RESULT_PATH}/{pair_name}.pdb"
        cmd3 = f"python {BASE_PATH}/example_scripts/merge-pdbs.py -i {PDBS_RESULT_PATH}/decode_pdb -o {output_pdb}"
        if not run_command(cmd3, f"Failed to run merge-pdbs.py: {pair_name}"):
            failed_systems.append(pair_name)
            continue
        
        # 4. Move results to final directory
        final_dir = f"{FINAL_RESULTS_PATH}/{pair_name}"
        os.makedirs(final_dir, exist_ok=True)
        
        # Copy contents from result directory to final directory
        for item in os.listdir(PDBS_RESULT_PATH):
            s = os.path.join(PDBS_RESULT_PATH, item)
            d = os.path.join(final_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        
        logging.info(f"Successfully processed PDB pair: {pair_name}")
        successful_systems.append(pair_name)
        
        # Clean result directory for next processing
        shutil.rmtree(PDBS_RESULT_PATH)
        os.makedirs(PDBS_RESULT_PATH, exist_ok=True)
    
    except Exception as e:
        # Error occurred while processing PDB pair
        logging.error(f"Error occurred while processing PDB pair {pair}: {str(e)}")
        failed_systems.append(pair.replace(';', '-'))

# Output final statistics
logging.info("\n===== Workflow completed =====")
logging.info(f"Successfully processed systems ({len(successful_systems)}): {', '.join(successful_systems)}")
logging.info(f"Failed systems ({len(failed_systems)}): {', '.join(failed_systems)}")
