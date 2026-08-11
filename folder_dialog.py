"""Native folder picker, run as its own subprocess (see app.py's browse_folder).

Runs standalone 
Prints the chosen path to stdout (empty line if cancelled).
"""
import os
import subprocess
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)

try:
    subprocess.run(
        ["osascript", "-e",
         f'tell application "System Events" to set frontmost of the first process whose unix id is {os.getpid()} to true'],
        capture_output=True, timeout=5,
    )
except Exception:
    pass  # not macOS, or osascript unavailable — dialog still opens, just may not be focused

folder = filedialog.askdirectory(title="Select a data folder")
root.destroy()
print(folder or "")
