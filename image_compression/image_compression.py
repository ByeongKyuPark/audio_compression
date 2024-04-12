import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write  # Import the write function

# Set plot parameters
plt.rcParams['figure.figsize'] = [16, 12]
plt.rcParams.update({'font.size': 18})

# Create a simple signal with two frequencies
dt = 0.001
t = np.arange(0, 1, dt)
f_clean = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)  # Sum of 2 frequencies
f_noisy = f_clean + 2.5 * np.random.randn(len(t))  # Add Gaussian noise

# Compute the FFT
n = len(t)
fhat = np.fft.fft(f_noisy, n)
PSD = np.abs(fhat) ** 2 / n  # Power spectrum density
freq = (1 / (dt * n)) * np.arange(n)
L = np.arange(1, np.floor(n / 2), dtype='int')  # Only plot the first half

# Use the PSD to filter out noise
indices = PSD > 100  # Find all freqs with larger power
PSDclean = PSD * indices  # Zero out all others
fhat = indices * fhat  # Zero out small Fourier coefficients
ffilt = np.fft.ifft(fhat)  # Inverse FFT for filtered time signal

# Convert the filtered signal back to the time domain and take the real part
filtered_signal = ffilt.real

# Save the signals to WAV files
sample_rate = int(1 / dt)  # Sample rate is the inverse of the time step
write('original_signal.wav', sample_rate, np.int16(f_clean / np.max(np.abs(f_clean)) * 32767))
write('noisy_signal.wav', sample_rate, np.int16(f_noisy / np.max(np.abs(f_noisy)) * 32767))
write('filtered_signal.wav', sample_rate, np.int16(filtered_signal / np.max(np.abs(filtered_signal)) * 32767))

# Plotting the Power Spectrum and Signals
fig, axs = plt.subplots(3, 1, figsize=(16, 12))

# Plot Original Clean and Noisy Signal
axs[0].plot(t, f_clean, color='k', linewidth=2, label='Clean')
axs[0].plot(t, f_noisy, color='c', linewidth=1.5, label='Noisy')
axs[0].set_title('Original Clean and Noisy Signal')
axs[0].legend()

# Plot Power Spectrum Density
axs[1].plot(freq[L], PSD[L], color='blue', linewidth=2)
axs[1].set_title('Power Spectrum Density')
axs[1].set_xlim(freq[L[0]], freq[L[-1]])

# Plot Filtered Signal
axs[2].plot(t, filtered_signal, color='red', linewidth=2, label='Filtered Signal')
axs[2].set_title('Filtered Signal')
axs[2].legend()

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
