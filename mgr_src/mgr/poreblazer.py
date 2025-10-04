import os
from pathlib import Path
from ase.io import write
import shutil
import subprocess
import logging

class PoreblazerRunner:
    def __init__(self, path_to_poreblazer, path_to_defaults, path_to_FF_atoms, path_to_bash='usr/bin/bash'):
        self.path_to_poreblazer = Path(path_to_poreblazer)
        self.path_to_defaults = Path(path_to_defaults)
        self.path_to_FF_atoms = Path(path_to_FF_atoms)
        self.path_to_bash = path_to_bash

    def write_atoms_to_xyz_and_dat_file(self, mat_atoms, path_to_output_folder=None):
        if path_to_output_folder is None:
            path_to_output_folder = Path.cwd()/mat_atoms.info['name'] 

        folder_path = Path(path_to_output_folder)
        folder_path.mkdir(parents=True, exist_ok=True)

        path_to_file = folder_path / Path(mat_atoms.info['name'] + '.xyz')
        if not path_to_file.exists():
            write(path_to_file, mat_atoms)

        material_name  = mat_atoms.info['name']
        lengths, angles = mat_atoms.get_cell_lengths_and_angles()[0:3], mat_atoms.get_cell_lengths_and_angles()[3:]
        path_to_dat = folder_path / Path(material_name+ '.dat')
        if not path_to_dat.exists():
            with open(path_to_dat, 'w') as f:
                f.write(material_name+ '.xyz\n')
                f.write(' '.join([str(l) for l in lengths]) + '\n')
                f.write(' '.join([str(a) for a in angles]) + '\n')

    def run_poreblazer_calculation(self, path_to_folder):
        path_to_folder = Path(path_to_folder)
        shutil.copy(self.path_to_defaults, path_to_folder)
        shutil.copy(self.path_to_FF_atoms, path_to_folder)

        executable_path = (self.path_to_poreblazer/'src/poreblazer.exe').resolve()
        logging.info(f"Executable path: {executable_path}")

        input_file_path = (path_to_folder/'input.dat').resolve()
        logging.info(f"Input file path: {input_file_path}")

        command = f"{executable_path} < {input_file_path}"
        logging.info(f"Command: {command}")

        subprocess.run([self.path_to_bash, '-c', command])