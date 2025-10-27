#!/usr/bin/env python3
"""
Motor Speed vs Sampling Frequency Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import read_data_results3 as rd

def analyze_file(filename):
    """Analyze a single data file to extract sampling frequency and motor speed."""
    results = rd.read_data3(filename)
    tv_sec = np.array(results[2])
    tv_usec = np.array(results[3])
    m_pos = np.array(results[5])
    m_stat = np.array(results[4])
    
    steady_state_mask = (m_stat == 16)
    
    tv_sec_filtered = tv_sec[steady_state_mask]
    tv_usec_filtered = tv_usec[steady_state_mask]
    m_pos_filtered = m_pos[steady_state_mask]
    
    time_filtered = tv_sec_filtered + tv_usec_filtered / 1e6
    
    dt_all = np.diff(time_filtered)
    dt_valid = dt_all[dt_all > 0]
    
    if len(dt_valid) > 0:
        median_dt = np.median(dt_valid)
        sampling_freq = 1.0 / median_dt
    else:
        sampling_freq = 0
    
    dp = np.diff(m_pos_filtered)
    instant_speeds = np.abs(dp / dt_all)
    motor_speed_value = np.median(instant_speeds)
    
    # Optical parameters
    wavelength = 546e-9  # m (green laser reference)
    lambda_min = 400e-9  # shortest wavelength expected (400 nm)
    metres_per_microstep = 3.74e-11  # from calibration
    optical_path_per_microstep = 2 * metres_per_microstep  # mirror movement
    
    # OPD velocity (m/s)
    v_opd = motor_speed_value * optical_path_per_microstep
    
    # Spatial sampling density (samples per meter of OPD)
    spatial_sample_density = sampling_freq / v_opd  # units: samples/m
    
    # Nyquist spatial frequency (cycles per meter)
    nyquist_freq = spatial_sample_density / 2.0
    
    # Max spatial frequency of signal (cycles per meter)
    max_spatial_freq = 2.0 / lambda_min
    
    # Nyquist criterion (must sample ≥ 2 points per cycle)
    nyquist_satisfied = nyquist_freq >= max_spatial_freq
    
    return {
        'sampling_freq': sampling_freq,
        'motor_speed': motor_speed_value,
        'v_opd': v_opd,
        'spatial_sample_density': spatial_sample_density,
        'nyquist_freq': nyquist_freq,
        'max_signal_freq': max_spatial_freq,
        'nyquist_satisfied': nyquist_satisfied
    }

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    files = [
        ("50k", "interferometry_data_laser_50k.txt"),
        ("100k", "interferometry_data_laser_100k.txt"),
        ("200k", "interferometry_data_laser_200k.txt")
    ]
    
    results = []
    for label, filename in files:
        filepath = os.path.join(script_dir, filename)
        if os.path.exists(filepath):
            result = analyze_file(filepath)
            result['label'] = label
            results.append(result)
    
    if len(results) < 2:
        print("ERROR: Not enough data files!")
        return
    
    motor_speeds = np.array([r['motor_speed'] for r in results])
    sampling_freqs = np.array([r['sampling_freq'] for r in results])
    labels = [r['label'] for r in results]
    
    slope, intercept = np.polyfit(motor_speeds, sampling_freqs, 1)
    fitted_values = slope * motor_speeds + intercept
    r_squared = 1 - (np.sum((sampling_freqs - fitted_values) ** 2) / 
                     np.sum((sampling_freqs - np.mean(sampling_freqs)) ** 2))
    
    print(f"Slope: {slope:.6f} Hz/(microsteps/s), R²: {r_squared:.4f}")
    
    # Plot 1: Motor Speed vs Sampling Frequency
    plt.figure(figsize=(10, 6))
    plt.scatter(motor_speeds, sampling_freqs, s=200, alpha=0.7, 
               c=['red', 'blue', 'green'])
    
    x_fit = np.linspace(motor_speeds.min() * 0.9, motor_speeds.max() * 1.1, 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, 'k--', linewidth=2)
    
    for i, label in enumerate(labels):
        plt.annotate(label, (motor_speeds[i], sampling_freqs[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=12)
    
    plt.xlabel('Motor Speed (microsteps/s)', fontsize=12)
    plt.ylabel('Sampling Frequency (Hz)', fontsize=12)
    plt.title('Motor Speed vs Sampling Frequency', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(script_dir, 'motor_speed_vs_sampling_freq.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Motor Speed vs Nyquist Frequency
    nyquist_freqs = np.array([r['nyquist_freq'] for r in results])
    max_signal_freq = results[0]['max_signal_freq']  # Same for all
    
    # Calculate linear fit for Nyquist frequency
    nyquist_slope, nyquist_intercept = np.polyfit(motor_speeds, nyquist_freqs, 1)
    nyquist_fitted = nyquist_slope * motor_speeds + nyquist_intercept
    nyquist_r_squared = 1 - (np.sum((nyquist_freqs - nyquist_fitted) ** 2) / 
                             np.sum((nyquist_freqs - np.mean(nyquist_freqs)) ** 2))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(motor_speeds, nyquist_freqs, s=200, alpha=0.7, 
               c=['red', 'blue', 'green'], label='Nyquist Frequency')
    
    # Plot linear fit
    x_fit_nyquist = np.linspace(motor_speeds.min() * 0.9, motor_speeds.max() * 1.1, 100)
    y_fit_nyquist = nyquist_slope * x_fit_nyquist + nyquist_intercept
    plt.plot(x_fit_nyquist, y_fit_nyquist, 'b-', linewidth=2, 
             label=f'Linear fit: y = {nyquist_slope:.2e}x + {nyquist_intercept:.2e}')
    
    # Add horizontal line for max signal frequency
    plt.axhline(y=max_signal_freq, color='r', linestyle='--', linewidth=2, 
                label=f'Max Signal Frequency ({max_signal_freq:.2e} cycles/m)')
    
    for i, label in enumerate(labels):
        plt.annotate(label, (motor_speeds[i], nyquist_freqs[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=12)
    
    plt.xlabel('Motor Speed (microsteps/s)', fontsize=12)
    plt.ylabel('Nyquist Frequency (cycles/m)', fontsize=12)
    plt.title('Motor Speed vs Nyquist Frequency', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Calculate maximum motor speed for given OPD range
    opd_range = 20e-3  # 10 mm in meters
    lambda_min = 400e-9  # nm
    max_spatial_freq = 1.0 / lambda_min
    metres_per_microstep = 3.74e-11
    optical_path_per_microstep = 2 * metres_per_microstep
    
    # At Nyquist limit: nyquist_freq = max_spatial_freq
    # nyquist_freq = (slope * motor_speed + intercept)
    # Solve for motor speed at intersection
    motor_speed_max = (max_spatial_freq - nyquist_intercept) / nyquist_slope
    
    # Calculate scanning time for 10 mm OPD range
    v_opd = motor_speed_max * optical_path_per_microstep
    scanning_time = opd_range / v_opd if v_opd > 0 else 0
    
    print(f"Max motor speed (Nyquist limit, R=1): {motor_speed_max:.0f} microsteps/s")
    print(f"Scanning time for {opd_range*1e3:.0f} mm OPD: {scanning_time:.1f} s")
    
    # Calculate motor speed for R = 2.5 (oversampling ratio)
    # R = spatial_sample_density / (2 * max_spatial_freq) where spatial_sample_density = sampling_freq / v_opd
    # For R=2.5, we need spatial_sample_density = 2.5 * 2 * max_spatial_freq = 5 * max_spatial_freq
    # But we want lower motor speed (more oversampling), so we solve from the relationship
    # At given motor speed, we want nyquist_freq = 2.5 * max_spatial_freq
    R = 2.5
    required_nyquist_freq = R * max_spatial_freq
    
    # Since we can't directly solve for lower motor speed, we find max motor speed that achieves R=1
    # then calculate what sampling frequency we'd need for R=2.5
    # R = 2.5 means we need 2.5x more sampling than R=1
    # So the required motor speed is lower by a factor of 2.5
    motor_speed_R = motor_speed_max / R
    
    # Recalculate for the lower speed
    v_opd_R = motor_speed_R * optical_path_per_microstep
    scanning_time_R = opd_range / v_opd_R if v_opd_R > 0 else 0
    
    print(f"Motor speed (R=2.5): {motor_speed_R:.0f} microsteps/s")
    print(f"Scanning time for {opd_range*1e3:.0f} mm OPD at R=2.5: {scanning_time_R:.1f} s")
    
    output_file2 = os.path.join(script_dir, 'motor_speed_vs_nyquist_freq.png')
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"Nyquist fit: slope={nyquist_slope:.2e}, R²={nyquist_r_squared:.4f}")
    plt.show()

if __name__ == "__main__":
    main()
