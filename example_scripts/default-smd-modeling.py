#!/usr/bin/env python3

import os
import argparse
import Xponge
import Xponge.forcefield.amber.ff14sb
import Xponge.forcefield.amber.tip3p
import Xponge.forcefield.amber.gaff as gaff
import subprocess

parser = argparse.ArgumentParser(description="Run sponge modeling with custom base directory")
parser.add_argument("script_dir", help="The base directory for the script")
args = parser.parse_args()
script_dir = args.script_dir

start_pdb_path = os.path.join(script_dir, "start.pdb")
end_pdb_path = os.path.join(script_dir, "end.pdb")
start_output_pdb_path = os.path.join(script_dir, "start-final.pdb")
end_output_pdb_path = os.path.join(script_dir, "end-final.pdb")
start_output_mol2_path = os.path.join(script_dir, "start-final.mol2")
end_output_mol2_path = os.path.join(script_dir, "end-final.mol2")
start_sponge_input_path = os.path.join(script_dir, "start")
end_sponge_input_path = os.path.join(script_dir, "end")

start = Xponge.load_pdb(start_pdb_path, ignore_hydrogen=True, ignore_unknown_name=True, ignore_seqres=False)
start.add_missing_residues()
start.add_missing_atoms()
Xponge.addSolventBox(start, WAT, 20) # 20A
# Blance the charge of the system
Xponge.Solvent_Replace(start, WAT, {K: 21 - int(round(start.charge)), CL: 21}) # Add K+ and Cl- ions
Xponge.save_sponge_input(start, start_sponge_input_path)
Xponge.save_pdb(start, start_output_pdb_path)
Xponge.save_mol2(start, start_output_mol2_path)

end = Xponge.load_pdb(end_pdb_path, ignore_hydrogen=True, ignore_unknown_name=True, ignore_seqres=False)
end.add_missing_residues()
end.add_missing_atoms()
Xponge.addSolventBox(end, WAT, 20) # 20A
# Blance the charge of the system
Xponge.Solvent_Replace(end, WAT, {K: 21 - int(round(end.charge)), CL: 21}) # Add K+ and Cl- ions
Xponge.save_sponge_input(end, end_sponge_input_path)
Xponge.save_pdb(end, end_output_pdb_path)
Xponge.save_mol2(end, end_output_mol2_path)

# Run minimization in the start and end directories
# Switch to the directory and run the sponge command
start_min_dir = os.path.join(script_dir, "start-min")
end_min_dir = os.path.join(script_dir, "end-min")
os.makedirs(start_min_dir, exist_ok=True)
os.makedirs(end_min_dir, exist_ok=True)
os.chdir(start_min_dir)
# Command: '/lustre/grp/gyqlab/chenzhenyu/zhouqk/sponge_v1.4/SPONGE/SPONGE' -mode minimization -step_limit 2000 -write_information_interval 1 -write_mdout_interval 1  -default_in_file_prefix ../start
subprocess.run([
    '/lustre/grp/gyqlab/chenzhenyu/zhouqk/sponge_v1.4/SPONGE/SPONGE',
    '-mode', 'minimization',
    '-step_limit', '2000',
    '-write_information_interval', '1',
    '-write_mdout_interval', '1',
    '-default_in_file_prefix', f'{script_dir}/start'
])
os.chdir(end_min_dir)
# Command: '/lustre/grp/gyqlab/chenzhenyu/zhouqk/sponge_v1.4/SPONGE/SPONGE' -mode minimization -step_limit 2000 -write_information_interval 1 -write_mdout_interval 1  -default_in_file_prefix ../end
subprocess.run([
    '/lustre/grp/gyqlab/chenzhenyu/zhouqk/sponge_v1.4/SPONGE/SPONGE',
    '-mode', 'minimization',
    '-step_limit', '2000',
    '-write_information_interval', '1',
    '-write_mdout_interval', '1',
    '-default_in_file_prefix', f'{script_dir}/end'
])

# Run the create-ref-cli.py script
os.chdir(script_dir)
# Run command: python '/lustre/grp/gyqlab/chenzhenyu/zhouqk/sponge_v1.4/create-ref-cli.py' -s start-final.pdb -e end-final.pdb -t end-final.mol2 -c end-min/restart_coordinate.txt -o .
subprocess.run([
    'python',
    '/lustre/grp/gyqlab/chenzhenyu/zhouqk/sponge_v1.4/create-ref-cli.py',
    '-s', start_output_pdb_path,
    '-e', end_output_pdb_path,
    '-t', end_output_mol2_path,
    '-c', os.path.join(end_min_dir, 'restart_coordinate.txt'),
    '-o', script_dir
])
