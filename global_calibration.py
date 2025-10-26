import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
import read_data_results3 as rd
import os

# Clean up previously generated plot files
script_dir = os.path.dirname(os.path.abspath(__file__))
for f in os.listdir(script_dir):
    if f.endswith("_filtered_signal.png"):
        os.remove(os.path.join(script_dir, f))

# Process only the green laser dataset
files = [os.path.join(script_dir, "interferometry_data_green_laser.txt")]

for file in files:
    base = os.path.splitext(os.path.basename(file))[0]
    try:
        # Read data
        results = rd.read_data3(file)
        x = np.array(results[5])  # Position
        y = np.array(results[1])  # ADC2 (Detector 2)

        # Normalize data
        y = y - np.mean(y)
        y_max = np.max(np.abs(y))
        if y_max > 0:
            y = y / y_max

        # Apply high-pass filter
        sos = signal.butter(2, 0.01, btype='highpass', output='sos')
        y = signal.sosfiltfilt(sos, y)

        # Remove filter artifacts
        percent_mask = 0.05
        y = y[int(len(y)*percent_mask):int(len(y)*(1-percent_mask))]
        x = x[int(len(x)*percent_mask):int(len(x)*(1-percent_mask))]
        x -= np.min(x)

        # Find zero-crossing points
        crossing_points = np.array([])
        for i in range(len(y)-1):
            if (y[i] <= 0 and y[i+1] >= 0) or (y[i] >= 0 and y[i+1] <= 0):
                # Linear interpolation to find exact crossing point
                xi, xj = x[i], x[i+1]
                yi, yj = y[i], y[i+1]
                
                if yj != yi:  # avoid divide-by-zero
                    crossing_x = xi - yi * (xj - xi) / (yj - yi)
                    if np.isfinite(crossing_x):
                        crossing_points = np.append(crossing_points, crossing_x)

        # Calculate distances between crossing points
        diff = np.array([])
        for i in range(len(crossing_points)-1):
            diff = np.append(diff, abs(crossing_points[i+1] - crossing_points[i]))

        # Calculate calibration for green laser dataset
        if len(diff) > 0:
            mean_diff = np.mean(diff)
            wl_ref = 532.0e-9  # reference wavelength (532 nm)
            metres_per_microstep = wl_ref / (2.0 * mean_diff)
            print(f"{base}: Mean crossing distance = {mean_diff:.1f} µsteps")
            print(f"{base}: CALIBRATION = {metres_per_microstep:.2e} m/µstep")
        else:
            print(f"{base}: No crossing points found")

        # Plot filtered signal with crossing points
        plt.figure(figsize=(8, 3))
        plt.plot(x, y, '-', linewidth=0.8, color='tab:blue')
        plt.axhline(0.0, color='k', linestyle='--', alpha=0.4)
        if len(crossing_points) > 0:
            plt.plot(crossing_points, np.zeros_like(crossing_points), 'ro', markersize=4, label=f'{len(crossing_points)} crossings')
        plt.xlabel("Position (µsteps)")
        plt.ylabel("Filtered Signal")
        plt.title(f"Filtered Signal — {base}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{base}_filtered_signal.png", dpi=150, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Error processing {base}: {e}")
        continue