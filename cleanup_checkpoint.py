"""
Simple script to clean up checkpoint file and start fresh.
"""
import os

CHECKPOINT_FILE = 'bulk_test_checkpoint.json'

if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print(f"Removed checkpoint file: {CHECKPOINT_FILE}")
    print("Next run will start from the beginning.")
else:
    print("No checkpoint file found.")
