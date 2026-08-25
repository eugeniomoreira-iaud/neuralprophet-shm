# Technique in Section 3: Spectral Analysis for Incomplete Data

The technique used in Section 3 to identify the periodicities in the compensated inclination signal is the **Lomb–Scargle periodogram**, supplemented by two robust validation controls (a **window function** and a **permutation null**) and a high-pass filtering strategy for the diurnal band. Additionally, a **Welch periodogram** is used as an independent secondary check.

## How it works:

### 1. The Lomb-Scargle Periodogram
Standard spectral analysis methods (like the ordinary periodogram) require a continuous grid of data points. Since this dataset is missing 31.5% of its hourly slots (with the majority of missing hours clustered in long gaps exceeding 30 days), interpolating these gaps would introduce artificial low-frequency power. 

Instead, the study uses the Lomb–Scargle periodogram. This method fits a sinusoid at each trial frequency via least squares using only the available (non-missing) observations. This calculates the spectrum directly on the incomplete data, consuming the gaps rather than needing to impute them first.

### 2. Validation Controls
To distinguish real physical periodicities from statistical noise or artefacts created by the missing data pattern, two controls are used:
*   **Window Function:** The same periodogram is applied to the *observation mask* (a binary series where 1 means data is present and 0 means missing) rather than the actual values. Any spectral peak appearing in this window function is a product of the outage pattern, not the physical structure. For example, an apparent 971-day peak in the data was identified as an artefact and disqualified because its power in the window function (0.1336) was larger than its power in the data (0.1178).
*   **Permutation Null:** The signal's values are randomly shuffled against the observation timestamps, destroying any real periodicity but perfectly preserving the exact sampling pattern. By generating 200 replicates, the study calculates the 95th and 99th percentiles of random noise to establish a threshold. Only peaks that clear the 99% permutation null (like the 370.1-day, 183.0-day, and a 113.9-day feature) are accepted as genuine.

### 3. Diurnal Band Analysis
To isolate high-frequency periodicities (the diurnal cycle and its harmonics), the series is first high-pass filtered by subtracting a 168-hour (7-day) rolling mean. 
*   This filtering process artificially injects a spurious peak near 7.28 days (the response peak of the filter). The study verifies this is an artefact by changing the rolling mean window to 720 hours and observing the peak shift.
*   The genuine daily peak (23.93 h) and its second (11.99 h) and third (8.00 h) harmonics emerge clearly.
    > **Understanding Harmonics:** In spectral analysis, physical daily cycles (like sunlight or temperature) rarely form perfect, smooth waves. Because they are often asymmetrical (e.g., a sharp peak at noon and a flat valley at night), the mathematics must build that shape by stacking multiple perfect waves on top of each other. 
    > * The **fundamental peak** (~24 hours) is the main daily cycle. 
    > * The **second harmonic** (~12 hours) and **third harmonic** (~8 hours) are waves that oscillate exactly two and three times as fast. Their presence doesn't mean a physical event happens every 12 or 8 hours; instead, they are the mathematical fractions required to "sculpt" the lopsided, real-world shape of the daily cycle.
*   As a final verification, a **Welch periodogram** is run purely on the longest unbroken block of data (where missing data is not an issue) to independently confirm the 1.0000-day daily peak.
    > **Welch's Method (Power Spectral Density Estimation):** Welch's periodogram is a non-parametric technique used to estimate the power spectra of a time-series signal. It improves upon the standard periodogram by drastically reducing the variance of the spectral estimate (at the cost of some frequency resolution). It achieves this through three steps:
    > 1. **Segmentation:** The continuous time-domain signal is divided into multiple overlapping blocks (segments).
    > 2. **Windowing:** A tapering function (such as a Hann or Hamming window) is applied to each segment to suppress spectral leakage caused by edge discontinuities.
    > 3. **Averaging:** The discrete Fourier transform (DFT) is computed for each windowed segment to obtain local periodograms. The final Power Spectral Density (PSD) is the ensemble average of all these individual periodograms.
    > 
    > Because the averaging process smooths out random statistical noise, genuine structural periodicities stand out more robustly. However, because it strictly requires evenly sampled, continuous data, it could only be applied to the longest uninterrupted sequence in this dataset.
