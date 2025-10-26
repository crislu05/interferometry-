###################################################################################################
### Reads all interferometry data files in this folder and displays Detector 2 vs position.
### Uses read_data_results3.read_data3() which skips header rows like 'ADC1 ADC2 ...'.
###################################################################################################

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import read_data_results3 as rd

# Force unbuffered output for better debugging
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

# ------------------------------------------------------------------------------------------------
# Locate all files whose names start with "interferometry_data_"
# ------------------------------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
files = [
    os.path.join(script_dir, f)
    for f in os.listdir(script_dir)
    if os.path.isfile(os.path.join(script_dir, f))
    and f.lower().startswith("interferometry_data_")
    and f.lower().endswith(".txt")
]
files.sort()

if not files:
    raise FileNotFoundError(
        'No files found whose names start with "interferometry_data_" in this folder.'
    )

print(f"[INFO] Found {len(files)} data files:\n")
for f in files:
    print("  -", os.path.basename(f))
print()

# ------------------------------------------------------------------------------------------------
# Read and plot each file
# ------------------------------------------------------------------------------------------------
for file in files:
    base = os.path.splitext(os.path.basename(file))[0]
    print(f"[INFO] Processing {base} ...")

    try:
        results = rd.read_data3(file)
    except Exception as e:
        print(f"  [WARNING] Skipped {base}: {e}")
        continue

    try:
        y2 = np.array(results[1], dtype=float)  # Detector 2 (ADC2)
        x  = np.array(results[5], dtype=float)  # Mirror position (M_POS)
    except Exception as e:
        print(f"  [WARNING] Column indexing failed for {base}: {e}")
        continue

    # ---- Plot Detector 2 only ----
    plt.figure(figsize=(7, 4))
    plt.plot(x, y2, 'o-', markersize=2, color='tab:blue')
    plt.xlabel("Position [μsteps]")
    plt.ylabel("Detector 2 Signal (a.u.)")
    plt.title(f"Interferogram — {base} (Detector 2)")
    plt.tight_layout()
    
    # Save plot instead of showing (avoids GUI blocking)
    output_path = os.path.join(script_dir, f"{base}_detector2_plot.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved plot to: {os.path.basename(output_path)}")

print("\n[INFO] All Detector 2 plots saved successfully.")
