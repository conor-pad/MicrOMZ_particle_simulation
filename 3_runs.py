#!/usr/bin/env python3
import subprocess
import sys
import shutil
import os
import glob

def run(command):
    print(f"Running: {' '.join(command)}")
    res = subprocess.run(command)
    if res.returncode != 0:
        print(f"Command failed with return code {res.returncode}")
        sys.exit(res.returncode)

def move_outputs(folder_name):
    os.makedirs(folder_name, exist_ok=True)
    # Move all generated media files into the specific run folder
    for ext in ['*.mp4', '*.png', '*.jpg']:
        for file in glob.glob(ext):
            shutil.move(file, os.path.join(folder_name, file))

if __name__ == '__main__':
    # Backup the original biopar.py
    shutil.copyfile('biopar.py', 'biopar_backup.py')

    khyd_vals = ['1e-2', '1e-1', '1e-3']

    try:
        for val in khyd_vals:
            print(f"\n--- Starting run with k_hyd_max = {val} ---")
            
            # Append the override to the bottom of biopar.py
            with open('biopar.py', 'a') as f:
                f.write(f'\nk_hyd_max = {val}\n')
            
            # Execute the model
            run([sys.executable, 'main.py'])
            
            # Move outputs to a dedicated folder to prevent overwriting
            move_outputs(f'output_khyd_{val}')

    finally:
        # Restore the original biopar.py so nothing is permanently changed
        shutil.copyfile('biopar_backup.py', 'biopar.py')
        os.remove('biopar_backup.py')