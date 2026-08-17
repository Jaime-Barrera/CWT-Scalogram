# CWT-Scalogram: Interactive Volatility Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Interactive web application developed in **Streamlit** for the generation of financial volatility scalograms using the **Continuous Wavelet Transform (CWT)**. The tool incorporates an adaptive **SUREShrink** (Stein's Unbiased Risk Estimate) denoising algorithm to isolate genuine volatility signals from market noise, enabling precise time-frequency analysis of financial assets.

---

## 📋 Key Features

- **Real-Time Data Extraction**: Automated retrieval of historical price data (stocks, cryptocurrencies, ETFs) via the Yahoo Finance API (`yfinance`).
- **Continuous Wavelet Transform (CWT)**: Decomposition of the log-return time series using complex wavelets (Morlet) to analyze volatility energy across multiple scales.
- **Adaptive Denoising (SUREShrink)**: Implementation of Stein's unbiased risk estimator to calculate optimal smoothing thresholds, minimizing the mean squared error (MSE). Includes a manual threshold option for advanced users.
- **Interactive Visualization**: Graphical representation of energy (scalogram) using dynamic heatmaps with **Plotly**, allowing zooming, panning, and exact value querying by coordinate.
- **Configurable Interface**: Sidebar panel to adjust the asset symbol (ticker), date range, maximum resolution scale, and filtering parameters.

---

## 📚 Theoretical Background

### 1. Continuous Wavelet Transform (CWT)
The CWT provides a time-frequency representation of the financial signal, overcoming the resolution limitations of the Short-Time Fourier Transform (STFT). It uses a mother wavelet (in this case, the complex Morlet wavelet `cmor1.5-1.0`) that is scaled and translated along the log-return series. This allows for high temporal resolution at high frequencies (short-term) and high frequency resolution at low scales (long-term).

### 2. SUREShrink Denoising
The thresholding method is based on the work of Donoho and Johnstone (1995) [*Adapting to Unknown Smoothness via Wavelet Shrinkage*]. The algorithm:
1. Estimates the noise level ($\sigma$) using the median absolute deviation of the wavelet coefficients.
2. Calculates the statistical risk (MSE) for different possible thresholds.
3. Selects the threshold that minimizes this risk (SURE).
4. Applies soft thresholding to the coefficients, preserving genuine volatility peaks while attenuating background noise. If data dispersion is extreme, the system safely falls back to Donoho's universal threshold.
