###################################################################################################
### Data cleaning: Remove abnormal status points (M_STAT filtering)
###################################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal.windows import tukey
import read_data_results3 as rd

# ---------- File discovery ----------
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
    raise FileNotFoundError("No interferometry_data_*.txt files found in this folder.")

print(f"[INFO] Found {len(files)} file(s):")
for f in files:
    print("  -", os.path.basename(f))
print()

# ---------- Helper function: Log-variance persistence filtering ----------
def filter_by_log_variance_persistence(pos, signal, window_size=100, tail_ratio=0.3, z=3.0, K=50):
    """
    Compute rolling variance s_t^2 over a centered window, take log-variance
    y_t = log(s_t^2), estimate steady target from the tail (last tail_ratio of points)
    using robust stats (median/MAD). Find the first index where y_t stays within
    [center ± z*scale] for K consecutive points, and trim everything before it.

    Returns:
        pos_filtered, signal_filtered, cutoff_index, (y_logvar, center, scale)
    """
    signal = np.asarray(signal)
    pos = np.asarray(pos)
    n = len(signal)
    if n == 0:
        return pos, signal, 0, (np.array([]), 0.0, 1.0)

    w = max(3, int(window_size))
    half = w // 2
    # Rolling variance (centered window with clamped edges)
    rolling_var = np.array([
        np.var(signal[max(0, i - half): min(n, i + half + 1)])
        for i in range(n)
    ], dtype=float)

    # Log-variance (stabilize scale); add tiny epsilon to avoid log(0)
    eps = 1e-12
    y_logvar = np.log(rolling_var + eps)

    # Central 30% robust statistics (35% to 65%)
    central_start = int(0.35 * n)
    central_end = int(0.65 * n)
    central = y_logvar[central_start:central_end]
    if len(central) < max(20, K):
        # Not enough central data; fall back to median/MAD of all
        central = y_logvar
    center = float(np.median(central))
    mad = float(np.median(np.abs(central - center)))
    scale = 1.4826 * (mad if mad > 0 else 1.0)

    lower = center - z * scale
    upper = center + z * scale

    in_band = (y_logvar >= lower) & (y_logvar <= upper)

    # Find start of control region (first K consecutive points in band)
    start_index = 0
    for i in range(0, n - K + 1):
        if np.all(in_band[i:i + K]):
            start_index = i
            break

    # Find end of control region (first point after start that goes out of band)
    end_index = n  # Default: keep all data after start
    if start_index < n:
        # Look for first out-of-band point after the control region starts
        for i in range(start_index, n):
            if not in_band[i]:
                end_index = i
                break
    
    # Apply both start and end filtering
    cutoff_index = start_index
    pos_filtered = pos[start_index:end_index]
    signal_filtered = signal[start_index:end_index]


    return pos_filtered, signal_filtered, cutoff_index, (y_logvar, center, scale)

# ---------- Helper function: Gaussian-modulated cosine curve fitting ----------
def fit_gaussian_cosine_curve(pos, signal):
    """
    Fit a Gaussian-modulated cosine:
        y = A + B * exp[-0.5 * (sigma_k * x)^2] * cos(k0*x - phi)

    Parameters:
    - pos: position array (μsteps)
    - signal: interferogram signal

    Returns:
    - (A, B, k0, sigma_k, phi): fitted parameters
    - (x_fit, y_fit): fitted curve for plotting
    - R^2 goodness-of-fit
    """
    def model(x, A, B, k0, sigma_k, phi):
        return A + B * np.exp(-0.5 * (sigma_k * x)**2) * np.cos(k0 * x - phi)

    # CRITICAL FIX: Convert position to physical OPD in meters
    metres_per_microstep = 1.94e-11 * 2.0  # round-trip OPD
    pos_m = np.asarray(pos) * metres_per_microstep  # convert to meters
    

    # Initial guesses with proper scaling
    A0 = np.mean(signal)
    B0 = (np.max(signal) - np.min(signal)) / 2
    opd_span = np.max(pos_m) - np.min(pos_m)
    k0_guess = 2 * np.pi / opd_span * 10  # frequency in rad/m
    sigma_guess = 1.0 / opd_span  # envelope scale ~ 1/OPD_span
    phi0 = 0

    p0 = [A0, B0, k0_guess, sigma_guess, phi0]

    # Bounds with proper scaling - make them much more generous
    bounds = (
        [A0 - 2*B0, 0, 0, 1e-6, -2*np.pi],
        [A0 + 2*B0, 3*B0, 1e8, 1e6, 2*np.pi]
    )
    
    # Check if initial guesses are within bounds and adjust if needed
    lower_bounds, upper_bounds = bounds
    for i, (p0_val, lb, ub) in enumerate(zip(p0, lower_bounds, upper_bounds)):
        if not (lb <= p0_val <= ub):
            if p0_val < lb:
                p0[i] = lb * 1.1
            elif p0_val > ub:
                p0[i] = ub * 0.9

    try:
        popt, pcov = curve_fit(model, pos_m, signal, p0=p0, bounds=bounds, maxfev=20000)
        A, B, k0, sigma_k, phi = popt

        # Generate fitted curve (convert back to μsteps for plotting)
        x_fit_m = np.linspace(np.min(pos_m), np.max(pos_m), 1000)
        x_fit = x_fit_m / metres_per_microstep  # convert back to μsteps for plotting
        y_fit = model(x_fit_m, A, B, k0, sigma_k, phi)

        # R²
        y_pred = model(pos_m, A, B, k0, sigma_k, phi)
        ss_res = np.sum((signal - y_pred)**2)
        ss_tot = np.sum((signal - np.mean(signal))**2)
        r_squared = 1 - ss_res/ss_tot

        # Derived physical quantities
        coherence_length = 1/sigma_k
        wavelength = 2*np.pi/k0  # meters
        delta_lambda = wavelength**2 * sigma_k / (2*np.pi)  # λ²Δk/(2π)
        print(f"    [INFO] Wavelength: {wavelength*1e9:.1f} nm, Coherence length: {coherence_length*1e6:.2f} μm, R² = {r_squared:.4f}")

        return (A, B, k0, sigma_k, phi), (x_fit, y_fit), r_squared

    except Exception as e:
        print(f"    [WARNING] Gaussian-cosine fitting failed: {e}")
        return None, None, 0.0

# ---------- Helper function: FFT spectrum analysis ----------
def compute_fft_spectrum(pos, signal, alpha=0.2):
    """
    Compute FFT spectrum to retrieve original wavelength spectrum from interferometric data.
    Uses the calibration approach from the existing codebase.
    
    Parameters:
    - pos: position array (microsteps)
    - signal: interferometric signal
    - alpha: Tukey window parameter (0=rectangular, 1=Hann)
    
    Returns:
    - wavelengths: wavelength array (nm)
    - spectrum: normalized spectrum amplitude
    - dominant_wavelength: dominant wavelength (nm)
    """
    # Use calibration from global calibration file (should give correct 532 nm)
    metres_per_microstep = 1.94e-11  # From global calibration file
    
    # Convert position to optical path difference (OPD)
    # Note: Factor of 2 for round-trip OPD as mentioned in global calibration
    opd = pos * 2.0 * metres_per_microstep  # round-trip OPD
    
    # Calculate OPD step size (dx) - CRITICAL for correct FFT
    dx = np.median(np.diff(opd))  # Use actual spacing, not resampled
    
    # Simple DC removal only (following the provided code approach)
    signal_centered = signal - np.mean(signal)
    
    # Optional: Apply windowing (can be disabled for direct demonstration)
    if alpha > 0:
        window = tukey(len(signal_centered), alpha=alpha)
        signal_windowed = signal_centered * window
    else:
        signal_windowed = signal_centered
    
    # Direct FFT computation (matching the provided code approach)
    fft_result = fftshift(fft(signal_windowed))
    frequencies = fftshift(fftfreq(len(opd), d=dx))  # spatial frequency in cycles/meter
    
    # Keep only positive frequencies
    pos_freq_mask = frequencies > 0
    pos_frequencies = frequencies[pos_freq_mask]  # cycles per meter
    pos_spectrum = np.abs(fft_result[pos_freq_mask])
    
    # Convert spatial frequency to wavelength
    # spatial_freq = 1/wavelength, so wavelength = 1/spatial_freq
    wavelengths = 1.0 / pos_frequencies  # meters
    wavelengths_nm = wavelengths * 1e9  # convert to nanometers
    
    # Filter out unrealistic wavelengths (focus on visible/UV range)
    reasonable_mask = (wavelengths_nm >= 200) & (wavelengths_nm <= 1000)
    wavelengths_nm = wavelengths_nm[reasonable_mask]
    pos_spectrum = pos_spectrum[reasonable_mask]
    
    # Normalize spectrum
    spectrum_normalized = pos_spectrum / np.max(pos_spectrum)
    
    # Find dominant wavelength
    dominant_idx = np.argmax(spectrum_normalized)
    dominant_wavelength = wavelengths_nm[dominant_idx]
    
    # Calculate spectral resolution diagnostics
    opd_span = opd[-1] - opd[0]  # Total OPD span
    lambda_center = 532e-9  # Expected wavelength
    theoretical_resolution = (lambda_center**2) / opd_span  # λ²/L formula
    nyquist_limit = lambda_center / 2.0  # Nyquist sampling limit
    
    print(f"    [INFO] FFT: Dominant wavelength: {dominant_wavelength:.1f} nm, Resolution: {theoretical_resolution*1e9:.3f} nm")
    
    return wavelengths_nm, spectrum_normalized, dominant_wavelength

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

        # --- Apply log-variance persistence filtering (tail median/MAD) ---
        pos_clean, adc2_clean, cutoff_idx, (y_logvar, ctr, scl) = filter_by_log_variance_persistence(
            pos_filtered, adc2_filtered,
            window_size=100,   # adjust ~3–10 periods if known
            tail_ratio=0.30,   # last 30% tail
            z=3.0,             # band width
            K=50               # persistence
        )
        
        print(f"  [INFO] Final clean data points: {len(pos_clean)}")
        
        # --- Fit Gaussian-cosine curve to filtered data ---
        if len(pos_clean) > 50:  # Need enough data points for fitting
            fit_params, fit_curve, r_squared = fit_gaussian_cosine_curve(pos_clean, adc2_clean)
            if fit_params is not None:
                print(f"  [INFO] Gaussian-cosine fit successful with R² = {r_squared:.4f}")
            else:
                print(f"  [WARNING] Gaussian-cosine fitting failed")
        else:
            print(f"  [WARNING] Insufficient data points for cosine fitting")
            fit_params, fit_curve, r_squared = None, None, 0.0
        
        # --- Compute FFT spectrum on fitted curve ---
        if fit_params is not None and fit_curve is not None:
            x_fit, y_fit = fit_curve
            wavelengths, spectrum, dominant_wavelength = compute_fft_spectrum(x_fit, y_fit)
        else:
            wavelengths, spectrum, dominant_wavelength = None, None, None
        
        # --- Plot results ---
        if fit_params is not None and fit_curve is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            # Top plot: Gaussian-cosine fitted data
            x_fit, y_fit = fit_curve
            ax1.plot(pos_clean, adc2_clean, 'o-', markersize=2, color='tab:blue', label='Filtered Data')
            ax1.plot(x_fit, y_fit, 'r--', linewidth=2, label=f'Gaussian-Cosine Fit (R²={r_squared:.3f})')
            
            
            ax1.set_xlabel("Position [μsteps]")
            ax1.set_ylabel("Detector 2 Signal (a.u.)")
            ax1.set_title(f"Gaussian-Cosine Fitted Interferometry Data — {os.path.basename(file)}")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Bottom plot: FFT spectrum
            if wavelengths is not None and spectrum is not None:
                ax2.plot(wavelengths, spectrum, 'b-', linewidth=1)
                ax2.axvline(x=dominant_wavelength, color='r', linestyle='--', 
                           label=f'Dominant: {dominant_wavelength:.1f} nm')
                ax2.set_xlabel("Wavelength (nm)")
                ax2.set_ylabel("Normalized Amplitude")
                ax2.set_title("FFT Spectrum (Original Wavelength)")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_xlim(200, 800)  # Extended range to show UV-visible
            else:
                ax2.text(0.5, 0.5, 'FFT Analysis Failed', transform=ax2.transAxes, 
                        ha='center', va='center', fontsize=12)
                ax2.set_title("FFT Spectrum (Analysis Failed)")
            
            plt.tight_layout()
            plt.show()
        else:
            print(f"  [WARNING] No cosine fit available for plotting")

    except Exception as e:
        print(f"[WARN] Skipped {os.path.basename(file)}: {e}")

print("\n[INFO] Data cleaning complete - abnormal status points removed.")