#!/usr/bin/python

import sys
import read_data_results3 as rd
import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import iminuit as im
import os
from scipy import signal
import scipy.fftpack as spf
import scipy.signal as sps
import scipy.interpolate as spi


# Step 1 get the data and the x position
script_dir = os.path.dirname(os.path.abspath(__file__))

# ---------- Cleanup: Remove previously generated files ----------
print("[INFO] Cleaning up previously generated files...")
cleanup_count = 0

for f in os.listdir(script_dir):
    full_path = os.path.join(script_dir, f)
    if os.path.isfile(full_path):
        # Remove all local calibration plot files (including debug plots)
        if (f.endswith("_local_calibration_spectrum.png") or 
            f.endswith("_crossing_points.png") or 
            f.endswith("_cubic_spline.png")):
            os.remove(full_path)
            cleanup_count += 1
            print(f"  [CLEANUP] Removed old plot: {f}")

print(f"[INFO] Cleanup complete - removed {cleanup_count} old files")

# Process both mercury datasets
mercury_files = [
    "interferometry_data_mercury_green_mercury_2.txt",
    "interferometry_data_mercury_green_yellowdoublet_2.txt"
]

for file_name in mercury_files:
    file = os.path.join(script_dir, file_name)
    print(f"Processing: {file_name}")
    
    try:
        results = rd.read_data3(file)
    except FileNotFoundError:
        print(f"File not found: {file_name}")
        continue

    # Step 1.1 - set the reference wavelength. Whatever units you use here will be theunits of your final spectrum
    lam_r = 546e-9 # units of m 
    
    # Apply M_STAT filtering (keep only M_STAT == 16)
    m_stat = np.array(results[4])  # M_STAT column
    mask = m_stat == 16
    print(f"  [INFO] M_STAT filtering: {np.sum(mask)}/{len(mask)} points kept")
    
    # Be carefull!!! Change for the correct detector by swapping one and zero here
    # y1 should contain the reference wavelength that needs correcting
    y1 = np.array(results[0])[mask]  # ADC1 (reference mercury green)
    y2 = np.array(results[1])[mask]  # ADC2 (broadband signal)
    pos = np.array(results[5])[mask]  # Position data
    
    #for now remove the mean, or remove the offset with a filter later
    #y1 = y1 - y1.mean()
    #y2 = y2 - y2.mean()

    # create a dummy x-axis (with just integers)
    # to figure out where the crossing points are
    # note, assumes that each x is regularly spaced within each wavelength
    x = np.arange(0, len(y1), 1)

    #step 2.1 butterworth filter to correct for misaligment (offset)
    # Use same filter parameters as global_calibration.py
    sos = signal.butter(2, 0.01, btype='highpass', output='sos')
    filtered = signal.sosfiltfilt(sos, y1)
    y1 = filtered
    filtered = signal.sosfiltfilt(sos, y2)
    y2 = filtered


    #step 2 get the x at which we cross
    crossing_pos = []
    for i in range(len(y1)-1):
        if (y1[i] <= 0 and y1[i+1] >= 0) or (y1[i] >= 0 and y1[i+1] <= 0) :
        #create the exact crossing point of 0 using correct linear interpolation
            xa, ya = x[i], y1[i]
            xb, yb = x[i+1], y1[i+1]
            if yb != ya:  # avoid divide-by-zero
                x_cross = xa - ya * (xb - xa) / (yb - ya)
                crossing_pos.append(x_cross)

    #step 3 shift the points

    k = 0
    x_corr_array = [0]
    last_pt = 0
    last_pt_corr = 0

    for period in range(len(crossing_pos)//2-1):
        # measured_lam is distance between 3 crossing points (e.g., [0,1,2]) = 1 wavelength
        measured_lam = crossing_pos[2*period+2] - crossing_pos[2*period]
        # define the x-scale using the reference wavelength (lam_r)
        shifting_ratio = lam_r/measured_lam
        while x[k]<crossing_pos[2*period+2]:
            x_corr = shifting_ratio*(x[k]-last_pt)+last_pt_corr
            x_corr_array.append(x_corr)
            k = k+1
        last_pt = x[k-1]
        last_pt_corr = x_corr_array[-1]
    x_corr_array = x_corr_array[1:]

    # ---------------------------------------------------------------------
    # Now apply the corrected x-axis from y1 to the interferogram from y2
    # ---------------------------------------------------------------------

    #step 4 create a uniform data set 
    #Cubic Spline part to map onto corrected onto regular grid
    xr = x_corr_array
    N = 1000000 # these are the number of points that you will resample - try changing this and look how well the resampling follows the data.
    xs = np.linspace(0, x_corr_array[-1], N)
    y = y2[:len(x_corr_array)]
    cs = spi.CubicSpline(xr, y)


    distance = xs[1:]-xs[:-1]

    # FFT to extract spectra
    yf1=spf.fft(cs(xs))
    dx = xs[1] - xs[0]  # Physical sample spacing in metres
    xf1=spf.fftfreq(len(xs), d=dx) # Correct frequency axis in cycles per metre
    xf1=spf.fftshift(xf1) #shifts to make it easier (google if interested)
    yf1=spf.fftshift(yf1)
    xx1=xf1[int(len(xf1)/2+1):len(xf1)]
    repx1=1.0/xx1  # Convert spatial frequency to wavelength: λ = 1/σ  

    # Extract wavelength and intensity arrays
    wavelengths = abs(repx1)
    intensities = abs(yf1[int(len(yf1)/2+1):len(yf1)])
    
    # Calculate baseline noise statistics (200-400 nm)
    baseline_mask = (wavelengths >= 200e-9) & (wavelengths <= 400e-9)
    baseline_mean = np.mean(intensities[baseline_mask])
    baseline_std = np.std(intensities[baseline_mask])
    
    # Known mercury spectral lines
    mercury_lines = {
        404.66e-9: "Hg violet",
        435.83e-9: "Hg blue",
        546.07e-9: "Hg green",
        576.96e-9: "Hg yellow",
        579.07e-9: "Hg yellow",
    }
    
    # Calculate SNR for each mercury peak
    peak_data = []
    snr_values = []
    for wavelength, label in mercury_lines.items():
        idx = np.argmin(np.abs(wavelengths - wavelength))
        peak_intensity = intensities[idx]
        snr = (peak_intensity - baseline_mean) / baseline_std if baseline_std > 0 else 0
        snr_values.append(snr)
        peak_data.append({'wavelength': wavelengths[idx], 'intensity': peak_intensity, 'snr': snr})
    
    # Calculate SNR statistics
    snr_min = np.min(snr_values)
    snr_max = np.max(snr_values)
    snr_mean = np.mean(snr_values)
    
    print(f"SNR Range: {snr_min:.1f} - {snr_max:.1f} (Mean: {snr_mean:.1f})")
    
    plt.figure("Fully corrected spectrum FFT", figsize=(10, 6))
    plt.title(f'Local Calibration FFT Spectrum - {file_name}\nAvg SNR = {snr_mean:.1f} (Range: {snr_min:.1f}-{snr_max:.1f})', fontsize=12)
    plt.plot(abs(repx1), abs(yf1[int(len(yf1)/2+1):len(yf1)]))
    
    # Plot mercury peaks with labels
    for peak in peak_data:
        plt.axvline(x=peak['wavelength'], color='red', linestyle='--', linewidth=1.5, alpha=0.6)
        plt.annotate(f"{peak['wavelength']*1e9:.1f} nm\nSNR: {peak['snr']:.1f}", 
                    xy=(peak['wavelength'], peak['intensity']), xytext=(10, 10),
                    textcoords='offset points', fontsize=8, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    plt.xlim(200e-9, 1300e-9)
    plt.xlabel('Wavelength (m)')
    plt.ylabel('Intensity (a.u.)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    base_name = os.path.splitext(file_name)[0]
    plot_file = os.path.join(script_dir, f"{base_name}_local_calibration_spectrum.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved plot to: {os.path.basename(plot_file)}")
