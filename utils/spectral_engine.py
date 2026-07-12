import numpy as np
import cv2
from scipy.fftpack import dct
import pywt

class SpectralAnalyzer:
    def __init__(self):
        pass

    def _get_radial_profile(self, magnitude_spectrum):
        """Compresses a 2D spectrum into a 1D line chart showing energy decay across concentric radii."""
        h, w = magnitude_spectrum.shape
        center = (h // 2, w // 2)
        y, x = np.indices((h, w))
        
        # Calculate distance of each pixel from the DC component center
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
        
        # Average the energy values at each distinct radius
        tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / np.maximum(nr, 1)
        
        return radial_profile[:min(h, w) // 2]

    def compute_fft_features(self, gray_img):
        """Computes 2D FFT, returning the visual heatmap and high-frequency statistical metrics."""
        f_transform = np.fft.fft2(gray_img)
        f_shift = np.fft.fftshift(f_transform)
        
        # Log-shifted magnitude spectrum for visualization and analysis
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)
        
        # Extract 1D energy decay curve
        radial_prof = self._get_radial_profile(magnitude_spectrum)
        
        # Isolate the high-frequency tail (outer 30% of the spectrum)
        split_idx = int(len(radial_prof) * 0.7)
        hf_tail = radial_prof[split_idx:]
        
        # Calculate the slope of the decay using linear regression
        x_axis = np.arange(len(radial_prof))
        slope, _ = np.polyfit(x_axis, radial_prof, 1)
        
        features = {
            "fft_hf_mean": float(np.mean(hf_tail)),
            "fft_hf_variance": float(np.var(hf_tail)),
            "fft_slope": float(slope)
        }
        return magnitude_spectrum, features

    def compute_dct_features(self, gray_img):
        """Computes 2D DCT and extracts high vs. low frequency energy ratios."""
        dct_2d = dct(dct(gray_img.T, norm='ortho').T, norm='ortho')
        
        h, w = dct_2d.shape
        # Low frequencies concentrate in the top-left corner
        low_freq = dct_2d[:h//4, :w//4]
        # High frequencies concentrate in the bottom-right corner
        high_freq = dct_2d[3*h//4:, 3*w//4:]
        
        lf_energy = np.sum(np.abs(low_freq)) + 1e-8
        hf_energy = np.sum(np.abs(high_freq))
        
        features = {
            "dct_hf_energy": float(hf_energy),
            "dct_energy_ratio": float(hf_energy / lf_energy)
        }
        # Log scale heatmap for visualization
        dct_heatmap = np.log(np.abs(dct_2d) + 1e-8)
        return dct_heatmap, features

    def compute_wavelet_features(self, gray_img):
        """Applies 2D Discrete Wavelet Transform using Daubechies ('db1') wavelet."""
        coeffs2 = pywt.dwt2(gray_img, 'db1')
        LL, (LH, HL, HH) = coeffs2
        
        # HH contains diagonal high frequencies where synthesis artifacts are most prominent
        features = {
            "wavelet_hh_mean": float(np.mean(np.abs(HH))),
            "wavelet_hh_std": float(np.std(HH)),
            "wavelet_energy": float(np.sum(HH**2))
        }
        return HH, features

    def extract_all(self, image_bgr):
        """Master method: Converts face crop to grayscale and executes all 3 spectral transforms."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        fft_map, fft_feats = self.compute_fft_features(gray)
        dct_map, dct_feats = self.compute_dct_features(gray)
        wavelet_map, wavelet_feats = self.compute_wavelet_features(gray)
        
        # Merge all statistical numbers into a single tabular feature dictionary
        combined_features = {**fft_feats, **dct_feats, **wavelet_feats}
        
        heatmaps = {
            "fft": fft_map,
            "dct": dct_map,
            "wavelet_hh": wavelet_map
        }
        
        return combined_features, heatmaps