import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz


def plot_filter_response(taps, fs, title="Filter Frequency Response"):
    # Calcola la risposta in frequenza (w è in radianti/campione)
    w, h = freqz(taps, worN=8000)

    # Converte w in Hz
    freq_hz = (w / np.pi) * (fs / 2)

    # Calcola la magnitudine in dB
    amplitude_db = 20 * np.log10(np.abs(h))

    # Calcola la fase in gradi
    phase_deg = np.unwrap(np.angle(h)) * 180 / np.pi

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Magnitudine
    ax1.plot(freq_hz, amplitude_db, 'b', label='Magnitude (dB)')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.grid(True)
    ax1.set_ylim([-100, 5])  # Limite tipico per vedere il ripple e la stopband

    # Plot Fase (su asse secondario)
    ax2 = ax1.twinx()
    ax2.plot(freq_hz, phase_deg, 'g--', label='Phase (degrees)')
    ax2.set_ylabel('Phase [deg]', color='g')

    plt.title(title)
    plt.show()