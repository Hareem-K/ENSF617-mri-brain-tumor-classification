# ============================================================
#  setup_colab.py — One-click Colab Setup Script
#  Brain Tumor MRI Classification Project
#
#  HOW TO USE:
#  Paste this into your FIRST Colab cell and run it:
#
#    from google.colab import drive
#    drive.mount('/content/drive')
#    exec(open('/content/drive/MyDrive/brain_tumor_classification/setup_colab.py').read())
#
#  After it finishes → Runtime → Restart session
#  Then run the verification cell (Cell 2 in your notebook)
# ============================================================

import os
import sys
import subprocess

def print_section(title):
    print("\n" + "=" * 54)
    print(f"  {title}")
    print("=" * 54)

def run_command(cmd):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0 and result.stderr.strip():
        print(f"  WARNING: {result.stderr.strip()[:200]}")
    return result.returncode

# ──────────────────────────────────────────────
#  Step 1 — Detect environment
# ──────────────────────────────────────────────
print_section("Step 1 — Detecting environment")

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

import platform
print(f"  Running in Colab : {IN_COLAB}")
print(f"  Python version   : {platform.python_version()}")
print(f"  OS               : {platform.system()} {platform.release()}")

if not IN_COLAB:
    print("  ERROR: This script is designed for Google Colab.")
    sys.exit(1)

# ──────────────────────────────────────────────
#  Step 2 — Set project path
# ──────────────────────────────────────────────
print_section("Step 2 — Setting up project path")

PROJECT_ROOT = "/content/drive/MyDrive/brain_tumor_classification"

if not os.path.exists(PROJECT_ROOT):
    print(f"  ERROR: Project folder not found at:\n  {PROJECT_ROOT}")
    print("\n  Please make sure you have:")
    print("  1. Mounted Google Drive")
    print("  2. Created the 'brain_tumor_classification' folder in Drive")
    print("  3. Uploaded all project files into that folder")
    sys.exit(1)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"  Project root : {PROJECT_ROOT}")

# ──────────────────────────────────────────────
#  Step 3 — Create folder structure
# ──────────────────────────────────────────────
print_section("Step 3 — Creating project structure")

folders = [
    "data/Training/glioma",
    "data/Training/meningioma",
    "data/Training/notumor",
    "data/Training/pituitary",
    "data/Testing/glioma",
    "data/Testing/meningioma",
    "data/Testing/notumor",
    "data/Testing/pituitary",
    "src",
    "experiments",
    "outputs/gradcam",
    "notebooks",
    "logs",
]

for folder in folders:
    os.makedirs(os.path.join(PROJECT_ROOT, folder), exist_ok=True)

print("  Folder structure created.")
print(f"\n  {PROJECT_ROOT}/")
for f in ["data/", "src/", "experiments/", "outputs/", "notebooks/", "logs/"]:
    print(f"    ├── {f}")

# ──────────────────────────────────────────────
#  Step 4 — Install dependencies
# ──────────────────────────────────────────────
print_section("Step 4 — Installing dependencies")

req_path = os.path.join(PROJECT_ROOT, "requirements.txt")

if not os.path.exists(req_path):
    print(f"  ERROR: requirements.txt not found at {req_path}")
    sys.exit(1)

print(f"  Installing from: {req_path}")
print("  This may take 2–3 minutes...\n")
run_command(f"pip install -q -r {req_path}")
print("\n  All dependencies installed.")

# ──────────────────────────────────────────────
#  Step 5 — Download dataset from Kaggle
# ──────────────────────────────────────────────
print_section("Step 5 — Dataset")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CLASSES  = ["glioma", "meningioma", "notumor", "pituitary"]

# Check if already downloaded
def count_images(folder):
    if not os.path.exists(folder):
        return 0
    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

already_downloaded = all(
    count_images(os.path.join(DATA_DIR, "Training", c)) > 100
    for c in CLASSES
)

if already_downloaded:
    print("  Dataset already present on Drive — skipping download.")
else:
    print("  Dataset not found. Setting up Kaggle credentials...")

    kaggle_dir  = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    drive_kaggle = "/content/drive/MyDrive/kaggle.json"

    os.makedirs(kaggle_dir, exist_ok=True)

    if os.path.exists(drive_kaggle):
        run_command(f"cp {drive_kaggle} {kaggle_json}")
        run_command(f"chmod 600 {kaggle_json}")
        print("  Kaggle credentials loaded from Drive.")
    else:
        print("\n  kaggle.json not found in your Drive root.")
        print("  Please upload kaggle.json to the root of Google Drive")
        print("  (not inside any folder), then run this script again.")
        sys.exit(1)

    print("\n  Downloading dataset (this takes 2–5 minutes)...")
    run_command(
        f"kaggle datasets download "
        f"-d masoudnickparvar/brain-tumor-mri-dataset "
        f"-p {DATA_DIR} --unzip"
    )
    print("  Dataset downloaded and extracted.")

# ──────────────────────────────────────────────
#  Step 6 — Verify dataset counts
# ──────────────────────────────────────────────
print_section("Step 6 — Dataset verification")

all_ok = True
for split in ["Training", "Testing"]:
    for cls in CLASSES:
        folder = os.path.join(DATA_DIR, split, cls)
        count  = count_images(folder)
        status = "OK" if count > 0 else "MISSING"
        if count == 0:
            all_ok = False
        print(f"  {split}/{cls:<15} {count:>5} images   {status}")

# ──────────────────────────────────────────────
#  Done
# ──────────────────────────────────────────────
print_section("Setup Complete")

if all_ok:
    print("""
  Everything is installed and your dataset is ready.

  IMPORTANT — do this now:
  ────────────────────────────────────────────────
  Go to  Runtime → Restart session
  ────────────────────────────────────────────────

  After restarting, run Cell 2 in your notebook
  to verify all libraries load correctly.
""")
else:
    print("""
  Some dataset folders appear empty.
  Please check your Kaggle credentials and run again.
""")
