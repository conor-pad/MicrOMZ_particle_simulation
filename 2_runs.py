#!/usr/bin/env python3
import subprocess
import sys
import shutil
import os

def run(command):
    print(f"Running: {' '.join(command)}")
    res = subprocess.run(command)
    if res.returncode != 0:
        print(f"Command failed with return code {res.returncode}")
        sys.exit(res.returncode)

if __name__ == '__main__':
    # 1. Create a backup of the original config
    shutil.copyfile('config.py', 'config_backup.py')

    try:
        # 2. First run: normal parameters
        run([sys.executable, 'main.py'])

        # 3. Append U_bg = 0 to the bottom of config.py (overriding previous values)
        with open('config.py', 'a') as f:
            f.write('\nU_bg = 0.0\n')

        # 4. Second run: static fluid (U_bg = 0)
        run([sys.executable, 'main.py'])

    finally:
        # 5. Restore the original config so nothing is permanently changed
        shutil.copyfile('config_backup.py', 'config.py')
        os.remove('config_backup.py')