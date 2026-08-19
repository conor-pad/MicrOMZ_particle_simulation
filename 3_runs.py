#!/usr/bin/env python3
import subprocess
import sys
import shutil
import os
import glob
import re

def set_config_value(filepath, var_name, value):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    updated = False
    for i, line in enumerate(lines):
        if line.strip().startswith(var_name) and '=' in line:
            indent = line[:len(line) - len(line.lstrip())]
            lines[i] = f"{indent}{var_name} = {value}\n"
            updated = True
            break
            
    if not updated:
        lines.append(f"{var_name} = {value}\n")
        
    with open(filepath, 'w') as f:
        f.writelines(lines)

def patch_loop_limit(filepath, limit_val):
    with open(filepath, 'r') as f:
        content = f.read()
    # Uses regex to find the safe_dt assignment and inject the new multiplier
    content = re.sub(r'(\.min\(\)\.item\(\)\s*\*\s*)[\d\.]+', rf'\g<1>{limit_val}', content)
    with open(filepath, 'w') as f:
        f.write(content)

def run(command):
    res = subprocess.run(command)
    if res.returncode != 0:
        sys.exit(res.returncode)

def move_outputs(folder_name):
    os.makedirs(folder_name, exist_ok=True)
    
    if os.path.exists('simulation_cache.pkl'):
        shutil.move('simulation_cache.pkl', os.path.join(folder_name, 'simulation_cache.pkl'))
        
    for d in glob.glob('Plots_T*'):
        if os.path.isdir(d):
            shutil.move(d, os.path.join(folder_name, d))

if __name__ == '__main__':
    shutil.copyfile('config.py', 'config_backup.py')
    shutil.copyfile('loop.py', 'loop_backup.py')

    runs = [
        {"limit": 0.69, "label": "limit_0.69"},
        {"limit": 0.90, "label": "limit_0.90"}
    ]

    try:
        for i, r in enumerate(runs, 1):
            folder_name = f"output_run_{i}_{r['label']}"
            
            shutil.copyfile('config_backup.py', 'config.py')
            shutil.copyfile('loop_backup.py', 'loop.py')
            
            # Applying your standard 14-day config parameters
            set_config_value('config.py', 'Total_Time', '14 * 86400')
            set_config_value('config.py', 'final_flush_duration', 1000.0)
            set_config_value('config.py', 'intermittent_physics_flush_time', 20.0)
            set_config_value('config.py', 'macro_cycle_time', 1000.0)
            
            # Patch the safe_dt limit in loop.py
            patch_loop_limit('loop.py', r["limit"])
            
            print(f"\n--- Starting Run {i}: {r['label']} ---")
            run([sys.executable, 'main.py'])
            move_outputs(folder_name)

    finally:
        # Restore backups 
        if os.path.exists('config_backup.py'):
            shutil.copyfile('config_backup.py', 'config.py')
            os.remove('config_backup.py')
        if os.path.exists('loop_backup.py'):
            shutil.copyfile('loop_backup.py', 'loop.py')
            os.remove('loop_backup.py')