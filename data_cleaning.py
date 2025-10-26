###################################################################################################
### Data cleaning: Remove abnormal status points (M_STAT filtering)
###################################################################################################

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from scipy.interpolate import interp1d
import read_data_results3 as rd

# ---------- Cleanup: Remove previously generated files ----------
script_dir = os.path.dirname(os.path.abspath(__file__))

print("[INFO] Cleaning up previously generated files...")
cleanup_count = 0

# Remove all filtered data files
for f in os.listdir(script_dir):
    full_path = os.path.join(script_dir, f)
    if os.path.isfile(full_path):
        # Remove filtered text files
        if f.lower().endswith("_filtered.txt"):
            os.remove(full_path)
            cleanup_count += 1
        # Remove filtered plot files
        elif "_filtered" in f.lower() and f.lower().endswith(".png"):
            os.remove(full_path)
            cleanup_count += 1

print(f"  ✓ Removed {cleanup_count} previously generated file(s)\n")

# ---------- File discovery ----------
files = [
    os.path.join(script_dir, f)
    for f in os.listdir(script_dir)
    if os.path.isfile(os.path.join(script_dir, f))
    and f.lower().startswith("interferometry_data_")
    and f.lower().endswith(".txt")
    and "_filtered" not in f.lower()  # Exclude filtered files
]
files.sort()
if not files:
    raise FileNotFoundError("No interferometry_data_*.txt files found in this folder.")

print(f"[INFO] Found {len(files)} original data file(s):")
for f in files:
    print("  -", os.path.basename(f))
print()

# ---------- Load null measurement for background subtraction ----------
null_file = None
null_mean = None
for f in files:
    if "null" in os.path.basename(f).lower():
        null_file = f
        break

if null_file:
    print(f"[INFO] Found null measurement: {os.path.basename(null_file)}")
    try:
        # Read null data
        null_cols = rd.read_data3(null_file)
        null_adc2 = np.array(null_cols[1], float)  # Detector 2 signal
        null_pos = np.array(null_cols[5], float)   # Mirror position
        
        # Calculate mean of null measurement
        null_mean = np.mean(null_adc2)
        print(f"  [INFO] Null measurement mean: {null_mean:.2f} a.u.")
    except Exception as e:
        print(f"  [WARN] Failed to load null measurement: {e}")
        null_mean = None
else:
    print("[WARN] No null measurement found - background subtraction will be skipped")
    null_mean = None

print()

# ---------- Main processing: Filter abnormal status points ----------
for file in files:
    try:
        print(f"[INFO] Processing {os.path.basename(file)} ...")

        # --- Read data ---
        cols = rd.read_data3(file)
        adc1 = np.array(cols[0], float)  # Detector 1
        adc2 = np.array(cols[1], float)  # Detector 2
        mstat = np.array(cols[4], float) # Mirror status
        pos = np.array(cols[5], float)   # Mirror position

        # --- Check if this is blue LED file (skip M_STAT filtering) ---
        is_blue_led = "blue_led" in os.path.basename(file).lower()
        
        if is_blue_led:
            # Use raw data without M_STAT filtering for blue LED
            adc2_filtered = adc2
            pos_filtered = pos
            print(f"  [INFO] Blue LED detected - using raw data (no M_STAT filtering)")
            print(f"  [INFO] Data points: {len(adc2_filtered)}")
            
            # Apply background subtraction for blue LED
            if null_mean is not None:
                adc2_filtered = adc2_filtered - null_mean
                print(f"  [INFO] Background subtracted (null mean: {null_mean:.2f} a.u.)")
        else:
            # --- Keep only steady velocity data (M_STAT == 16) ---
            mask = (mstat == 16)
            filtered_count = np.sum(mask)
            
            if not np.any(mask):
                print("  [WARN] No M_STAT==16 data found; skipping file.")
                continue
                
        # Apply filter
        adc2_filtered = adc2[mask]
        pos_filtered = pos[mask]

        print(f"  [INFO] Filtered data points: {filtered_count}")
        
        # Apply background subtraction (skip for null measurement itself)
        if null_mean is not None and "null" not in os.path.basename(file).lower():
            adc2_filtered = adc2_filtered - null_mean
            print(f"  [INFO] Background subtracted (null mean: {null_mean:.2f} a.u.)")
        elif "null" in os.path.basename(file).lower():
            print(f"  [INFO] Skipping background subtraction for null measurement")

        # --- Save filtered data to new file ---
        base = os.path.splitext(os.path.basename(file))[0]
        output_file = os.path.join(script_dir, f"{base}_filtered.txt")
        
        with open(output_file, 'w') as f:
            # Write header
            f.write("ADC1 ADC2 tv_sec tv_usec M_STAT M_POS LVDT_DL LVDT actual_speed\n")
            # Write filtered data
            if is_blue_led:
                # For blue LED, write all raw data
                for i in range(len(pos_filtered)):
                    f.write(f"{adc1[i]:.0f} {adc2_filtered[i]:.0f} 0 0 {mstat[i]:.0f} {pos_filtered[i]:.0f} 0 0 0\n")
            else:
                # For other files, write M_STAT filtered data
                for i in range(len(pos_filtered)):
                    f.write(f"{adc1[mask][i]:.0f} {adc2_filtered[i]:.0f} 0 0 16 {pos_filtered[i]:.0f} 0 0 0\n")
        
        print(f"  ✓ Saved filtered data to: {os.path.basename(output_file)}")

        # --- Compute FFT spectrum ---
        # Convert position to optical path difference (OPD)
        metres_per_microstep = 3.74e-11  # updated calibration factor from green laser analysis
        opd = pos_filtered * metres_per_microstep  # meters
        
        # Calculate actual step size from the data
        if len(opd) > 1:
            dx = np.mean(np.diff(opd))  # Use mean instead of median for more stable calculation
        else:
            print("  [WARN] Not enough data points for FFT analysis")
            continue
        
        print(f"  [DEBUG] OPD step size: {dx:.2e} meters")
        print(f"  [DEBUG] OPD range: {np.min(opd):.2e} to {np.max(opd):.2e} meters")
        
        # Resample to uniform OPD spacing for accurate FFT
        opd_uniform = np.linspace(opd.min(), opd.max(), len(opd))
        signal_uniform = interp1d(opd, adc2_filtered, kind='linear')(opd_uniform)
        print(f"  [DEBUG] Resampled to uniform OPD spacing for FFT accuracy")
        
        # Remove DC component and recenter the interferogram
        signal_centered = signal_uniform - np.mean(signal_uniform)
        
        # Find the zero-OPD burst (center of interferogram) and recenter
        center_idx = np.argmax(np.abs(signal_centered))
        signal_centered = np.roll(signal_centered, len(signal_centered)//2 - center_idx)
        
        print(f"  [DEBUG] Zero-OPD burst found at index: {center_idx}, recentered to center")
        
        # Compute FFT on recentered signal
        fft_result = fft(signal_centered)  # Don't use fftshift yet
        frequencies = fftfreq(len(opd_uniform), d=dx)  # spatial frequency in cycles/meter
        
        # Apply fftshift to center the spectrum
        fft_result_shifted = fftshift(fft_result)
        frequencies_shifted = fftshift(frequencies)
        
        # Keep only positive frequencies (right half of shifted spectrum)
        pos_freq_mask = frequencies_shifted > 0
        pos_frequencies = frequencies_shifted[pos_freq_mask]
        pos_spectrum = np.abs(fft_result_shifted[pos_freq_mask])
        
        # Convert spatial frequency to wavelength
        wavelengths = 1.0 / pos_frequencies  # meters
        wavelengths_nm = wavelengths * 1e9  # nanometers
        
        # Filter to reasonable wavelength range (200-1000 nm)
        reasonable_mask = (wavelengths_nm >= 200) & (wavelengths_nm <= 1000)
        wavelengths_nm = wavelengths_nm[reasonable_mask]
        pos_spectrum = pos_spectrum[reasonable_mask]
        
        # Normalize spectrum
        if np.max(pos_spectrum) > 0:
            spectrum_normalized = pos_spectrum / np.max(pos_spectrum)
        else:
            print("  [WARN] Zero spectrum amplitude detected")
            spectrum_normalized = pos_spectrum
        
        # Find dominant wavelength
        dominant_idx = np.argmax(spectrum_normalized)
        dominant_wavelength = wavelengths_nm[dominant_idx]
        
        print(f"  [INFO] FFT Analysis - Dominant wavelength: {dominant_wavelength:.1f} nm")
        
        # Skip plotting for null measurements
        if "null" in os.path.basename(file).lower():
            print(f"  [INFO] Skipping plot generation for null measurement")
            continue
        
        # --- Plot filtered data and FFT ---
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16))
        
        # Top plot: ADC1 (Detector 1)
        ax1.plot(pos_filtered, adc1[mask], 'o-', markersize=2, color='tab:red')
        ax1.set_xlabel("Position [μsteps]")
        ax1.set_ylabel("Detector 1 Signal (a.u.)")
        ax1.set_title(f"Detector 1 Signal — {os.path.basename(file)}")
        ax1.grid(True, alpha=0.3)
        
        # Second plot: ADC2 (Detector 2)
        ax2.plot(pos_filtered, adc2_filtered, 'o-', markersize=2, color='tab:blue')
        ax2.set_xlabel("Position [μsteps]")
        ax2.set_ylabel("Detector 2 Signal (a.u.)")
        
        # Set title based on whether M_STAT filtering was applied
        if is_blue_led:
            ax2.set_title(f"Raw Interferometry Data (No M_STAT filtering) — {os.path.basename(file)}")
        else:
            ax2.set_title(f"Filtered Interferometry Data (M_STAT==16) — {os.path.basename(file)}")
        
        ax2.grid(True, alpha=0.3)
        
        # Third plot: FFT spectrum (linear scale)
        ax3.plot(wavelengths_nm, spectrum_normalized, 'b-', linewidth=1.5)
        ax3.axvline(x=dominant_wavelength, color='r', linestyle='--', linewidth=2,
                   label=f'Dominant: {dominant_wavelength:.1f} nm')
        ax3.set_xlabel("Wavelength (nm)")
        ax3.set_ylabel("Normalized Amplitude")
        ax3.set_title("FFT Spectrum (Linear Scale)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(200, 800)  # Focus on visible/UV range
        
        # Bottom plot: FFT spectrum (log scale)
        ax4.semilogy(wavelengths_nm, spectrum_normalized + 1e-10, 'g-', linewidth=1.5)  # Add small offset to avoid log(0)
        ax4.axvline(x=dominant_wavelength, color='r', linestyle='--', linewidth=2,
                   label=f'Dominant: {dominant_wavelength:.1f} nm')
        ax4.set_xlabel("Wavelength (nm)")
        ax4.set_ylabel("Normalized Amplitude (Log Scale)")
        ax4.set_title("FFT Spectrum (Log Scale)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(200, 800)  # Focus on visible/UV range
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(script_dir, f"{base}_filtered_plot.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved plot to: {os.path.basename(plot_file)}")

    except Exception as e:
        print(f"[WARN] Skipped {os.path.basename(file)}: {e}")

print("\n[INFO] Data cleaning complete - abnormal status points removed.")