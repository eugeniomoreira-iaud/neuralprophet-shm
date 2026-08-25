# Observations and Proxy Comparison Explanations

This document stores detailed explanations regarding the analyses found in the proxy comparison report.

## 1. The Winter vs. Summer Regime in Solar Radiation (Figure 1)

In the analysis of Figure 1 and the surrounding clock alignment tests, the solar radiation data is deliberately split into a "winter regime" and a "summer regime". There are two primary reasons for this—one technical, and one physical:

### A. The Technical Reason: The Daylight Saving Time (DST) Mismatch
The most critical reason the analysis forces a split between seasons is how the different systems track time. 
*   **The Proxies (ERA5 & Ground Station):** Datasets like ERA5 and professional meteorological stations typically record data in strict **UTC** (Coordinated Universal Time) or a fixed standard time. Their clocks never jump forward or backward.
*   **The On-Structure Logger:** The local datalogger on the structure, however, was likely set to local Italian time, which observes **Daylight Saving Time (DST)**. In the spring, its clock jumps forward by one hour, and in the autumn, it falls back.

Because of this, the "true" solar noon (when the sun is highest in the sky) appears to happen at different hours on the logger's clock depending on the time of year. If the analysis did not split the year into summer and winter, the solar peak would be artificially "smeared" across two different hours, ruining the cross-correlation alignment. 

By splitting the analysis into seasons, the code perfectly isolates this signature: the table shows an exact **0-hour shift in the winter** and a **1-hour shift in the summer** between the proxies and the structure's air temperature (`tair`).

### B. The Physical Reason: Solar Amplitude and Daylight Width
Physically, the actual profile of solar radiation changes drastically between the two regimes:
*   **Summer Regime:** The sun reaches a much higher elevation angle in the sky, resulting in a significantly higher peak intensity at solar noon (often exceeding 800-900 W/m²). The curve is also much wider at the base because the sun rises earlier and sets later, creating a longer daylight window.
*   **Winter Regime:** The sun remains lower on the horizon. The peak intensity at noon is much weaker (often around 300-400 W/m²), and the bell curve is narrower due to the shorter days.

### Summary
When looking at Figure 1, splitting the regimes allows the mathematical alignment algorithm to lock onto the sharpest, cleanest daily peak without it being blurred by a 1-hour DST clock jump, while also respecting the fact that the actual volume of sunlight is completely different between the seasons.

## 2. Why the On-Structure Solar Radiation Differs from Proxies in Winter but Matches in Summer

If you examine the diurnal solar curves (like those in Figure 1), you'll notice the on-structure (`str`) solar radiation channel looks unrecognizable compared to the proxies (ERA5 and the ground station) during the winter, but appears much more similar in the summer. 

This happens because the on-structure sensor is severely flawed, and the winter environment makes those flaws mathematically devastating.

### A. The Midnight Noise Defect (Signal-to-Noise Ratio)
The on-structure radiation sensor is objectively malfunctioning. The report notes that it fails its own consistency checks and famously **reads roughly 60 W/m² at midnight**, when it should be zero. 
*   **In Winter:** The true peak of solar radiation during a winter day is very weak (because the sun is low). Therefore, a constant, broken baseline noise of +60 W/m² represents a massive percentage of the total daily signal. This noise completely overwhelms the real data, twisting the winter curve into a shape that looks nothing like the healthy proxies. In fact, the noise is so bad in winter that the mathematical algorithm struggles to even find solar noon, resulting in an interquartile range of 5.3 hours (it's essentially guessing).
*   **In Summer:** The true peak of solar radiation is massive (often exceeding 800 W/m²). The broken +60 W/m² baseline noise is still there, but because the real solar signal is so powerful, the noise is mathematically drowned out. The sheer volume of sunlight forces the curve to look roughly like a normal solar curve, making it match the healthy proxies much better.

### B. Shadowing and Solar Elevation Angle
The physical environment also plays a major role:
*   **In Winter:** The sun traces a very low arc across the southern sky. The ground station and the ERA5 grid model represent unobstructed, open-sky measurements. However, the on-structure sensor is mounted on a building. At low winter sun angles, the sensor is almost certainly cast into the shadows of nearby structures, parapets, or the building itself for large portions of the day. 
*   **In Summer:** The sun travels almost directly overhead. It easily clears any surrounding obstacles that were casting winter shadows. Thus, in the summer, the on-structure sensor finally "sees" the same unobstructed, open sky that the proxies see year-round.

**Summary:** In winter, the on-structure sensor is heavily shadowed and recording a weak signal that is completely swallowed by its own 60 W/m² hardware malfunction. In summer, the sun clears the shadows and blasts the sensor with enough raw energy to overpower the hardware defect, forcing the data to resemble the clean proxy datasets.

## 3. The Non-Zero Night-time Floor (Why Winter Radiation Doesn't Hit Zero at Night)

You are absolutely correct, and this is a fantastic observation! If you look closely at Figure 1, the night-time solar radiation (between 0:00 and 5:00 AM) behaves completely differently depending on the season:
*   **In Summer:** The night-time value drops down to a normal **~3 W/m²** (essentially zero, which is correct for pitch black).
*   **In Winter:** The night-time value hovers at a massive **~214 W/m²**. 

*(Note: The report mentions "roughly 60 W/m² at midnight"—this is the **overall** yearly average, but you've correctly spotted that this noise is entirely concentrated in the winter!)*

### What is this 214 W/m² value?
It is a severe hardware defect known as a **zero-offset error** or **dark signal**, but the fact that it only happens in winter points to a specific physical cause. Pyranometers generate a tiny electrical voltage when exposed to light. When it's dark, they should generate 0.00 mV. The datalogger is recording a "phantom" voltage at night in the winter due to one of the following likely culprits:

1. **Cold-Weather Electronics Failure:** The amplifier or Analog-to-Digital Converter (ADC) inside the datalogger is failing at low temperatures. When the winter air drops near freezing at night, the components drift wildly, generating a massive 214 W/m² phantom voltage. In the warm summer nights, the electronics behave normally (reading ~3 W/m²).
2. **An Internal Defroster/Heater Short:** Professional pyranometers often have internal heaters designed to melt frost and dew on cold winter nights. If there is a grounding issue, a short circuit, or electrical "crosstalk" inside the cable, the electrical power sent to the heater can leak directly into the sensor's delicate signal wire. Because the heater only turns on during freezing winter nights, you only see the 214 W/m² phantom spike in the winter.
3. **Artificial Lighting:** There is a powerful security light or streetlamp nearby that shines on the sensor. Because winter nights are much longer (and potentially the light is on a timer that only activates in winter months), the sensor picks it up. *(However, 214 W/m² is extremely bright—equivalent to about 20% of the full noon sun—making an electrical failure much more likely).*

**In conclusion:** The 214 W/m² is not real sunlight; it is a massive electrical or temperature-dependent hardware failure that ruins the entire winter dataset. Because this 214 W/m² noise is almost as strong as the real winter sun, the mathematical algorithm cannot even find the daily solar peak, causing the whole winter curve to look completely unlike the clean proxy data.

## 4. Seasonal Periods and Observation Counts (Is there enough data?)

When separating the data into "winter" and "summer" regimes, the analysis uses strict meteorological definitions to maximize the contrast in solar elevation and daylight hours:
*   **Winter:** December, January, and February.
*   **Summer:** June, July, and August.
*(Spring and Autumn are deliberately excluded from this specific diurnal clock test to ensure the mathematical peaks are as sharp and distinct as possible).*

### How many full daily cycles are used?
You correctly noted that the local on-structure (`str`) pyranometer does not cover the entire 8-year dataset. It only exists in the "current era" (from February 2025 to August 2026). Despite this shorter lifespan, there are still plenty of observations to form a robust statistical mean. 

Here are the exact number of full, valid 24-hour cycles (`n_days`) that were fed into the mean calculations for each source:

*   **On-Structure (`str`)**:
    *   Winter: **87 days**
    *   Summer: **115 days**
*   **Ground Station (`gs`)**:
    *   Winter: **676 days**
    *   Summer: **714 days**
*   **ERA5 (`era5`)**:
    *   Winter: **772 days**
    *   Summer: **799 days**

### Is 87 days enough?
Yes. Statistically, generating a mean diurnal curve (a 24-hour profile) stabilizes very quickly. A sample size of 87 full cycles is more than enough data to smooth out day-to-day weather variations (like passing clouds) and reveal the underlying baseline behavior—including the glaring 214 W/m² hardware defect!
