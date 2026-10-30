import subprocess
import sys
from pathlib import Path


folder = Path(__file__).resolve().parent
for script in ["problem1.py", "problem2.py", "problem3.py"]:
    subprocess.run([sys.executable, str(folder / script)], cwd=folder, check=True)
