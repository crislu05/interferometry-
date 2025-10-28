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
from scipy.ndimage import gaussian_filter1d


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
            f.endswith("_calibrated_spectrum.png") or
            f.endswith("_crossing_points.png") or 
            f.endswith("_cubic_spline.png")):
            os.remove(full_path)
            cleanup_count += 1
            print(f"  [CLEANUP] Removed old plot: {f}")

print(f"[INFO] Cleanup complete - removed {cleanup_count} old files")

# Load null measurement for baseline subtraction
null_file = os.path.join(script_dir, "interferometry_data_null (1).txt")
null_baseline_mean = None

if os.path.exists(null_file):
    print(f"[INFO] Loading null measurement for baseline subtraction...")
    try:
        null_results = rd.read_data3(null_file)
        null_m_stat = np.array(null_results[4])
        null_mask = null_m_stat == 16
        print(f"  [INFO] Null M_STAT filtering: {np.sum(null_mask)}/{len(null_mask)} points kept")
        
        # Get null interferogram (ADC2)
        null_y2 = np.array(null_results[1])[null_mask]
        
        # Calculate mean of null interferogram
        null_baseline_mean = np.mean(null_y2)
        print(f"  ✓ Null baseline mean (interferogram): {null_baseline_mean:.6e}")
    except Exception as e:
        print(f"  [WARN] Failed to load null measurement: {e}")
        null_baseline_mean = None
else:
    print(f"[WARN] Null measurement file not found, skipping baseline subtraction")

print()

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
    
    # Subtract null baseline mean if available (constant offset removal)
    if null_baseline_mean is not None:
        y = y - null_baseline_mean
        print(f"  [INFO] Subtracted null baseline mean from interferogram: {null_baseline_mean:.6e}")
    
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
    intensities_raw = abs(yf1[int(len(yf1)/2+1):len(yf1)])
    
    # Calculate baseline mean from 200-400 nm region on unsmoothed data
    baseline_mask = (wavelengths >= 200e-9) & (wavelengths <= 400e-9)
    baseline_mean = np.mean(intensities_raw[baseline_mask])
    baseline_std = np.std(intensities_raw[baseline_mask])
    print(f"  [INFO] Baseline mean (200-400 nm): {baseline_mean:.6e}")
    print(f"  [INFO] Baseline std (200-400 nm): {baseline_std:.6e}")
    
    # Apply Gaussian smoothing to FFT spectrum
    sigma = 1.0  # Adjust this parameter to control smoothing (higher = more smoothing)
    intensities = gaussian_filter1d(intensities_raw, sigma=sigma)
    print(f"  [INFO] Applied Gaussian smoothing to FFT spectrum (sigma={sigma})")
    
    # Detect peaks using scipy find_peaks on smoothed data
    peak_height = baseline_mean + 3 * baseline_std  # Minimum peak height (3 sigma above baseline)
    peak_distance = 10 # Minimum distance between peaks (in samples)
    
    peaks, properties = sps.find_peaks(intensities, height=peak_height, distance=peak_distance)
    
    peak_wavelengths = wavelengths[peaks]
    peak_intensities = intensities[peaks]
    
    # Calculate SNR for each peak and filter by SNR >= 10
    peak_snr_values = []
    if len(peaks) > 0:
        for peak_int in peak_intensities:
            snr = (peak_int - baseline_mean) / baseline_std if baseline_std > 0 else 0
            peak_snr_values.append(snr)
        
        # Filter peaks with SNR >= 10
        snr_threshold = 10
        mask_high_snr = np.array(peak_snr_values) >= snr_threshold
        
        peaks_filtered = peaks[mask_high_snr]
        peak_wavelengths_filtered = peak_wavelengths[mask_high_snr]
        peak_intensities_filtered = peak_intensities[mask_high_snr]
        peak_snr_filtered = np.array(peak_snr_values)[mask_high_snr]
        
        print(f"  [INFO] Detected {len(peaks)} peaks total")
        print(f"  [INFO] Filtered to {len(peaks_filtered)} peaks with SNR >= {snr_threshold}")
        if len(peaks_filtered) > 0:
            print(f"  [INFO] Peak details (SNR >= {snr_threshold}):")
            for i, (wl, inten, snr) in enumerate(zip(peak_wavelengths_filtered, peak_intensities_filtered, peak_snr_filtered)):
                print(f"    Peak {i+1}: {wl*1e9:.2f} nm, Intensity: {inten:.6e}, SNR: {snr:.2f}")
        
        # Update variables to use filtered peaks
        peaks = peaks_filtered
        peak_wavelengths = peak_wavelengths_filtered
        peak_intensities = peak_intensities_filtered
        peak_snr_values = peak_snr_filtered.tolist()
    else:
        print(f"  [INFO] No peaks detected")
    
    plt.figure("Fully corrected spectrum FFT", figsize=(10, 6))
    plt.title(f'Local Calibration FFT Spectrum - {file_name}', fontsize=12)
    plt.plot(wavelengths, intensities, 'b-', linewidth=1, label='Spectrum')
    if len(peaks) > 0:
        plt.scatter(peak_wavelengths, peak_intensities, color='r', s=100, marker='x', 
                   linewidths=2, zorder=5)
    plt.legend()
    
    plt.xlim(350e-9, 1100e-9)  # Extended range to show visible through near-IR lines
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