# Video Script


## Introduction

Hello, my name is Shubh Khare. In this project, I developed a weather-data quality checker for the Ridgeline Schools Weather Network.

The dataset contains readings from eight weather stations, collected every ten minutes between March and April 2025.

My objective was to identify unreliable readings, understand possible causes, and decide which data should be kept, corrected, or excluded.

## Confident call: S06 logger clock

Here, I am running the checker on the provided dataset.

The program successfully loaded all eight stations and generated 23 flags. Each flag contains information about a potential issue, the affected station and measurement, and the recommended action.

The output contains three types of decisions: keep, correct, and exclude.

There are eight keep decisions, three correction decisions, and twelve exclusion decisions.

An important example is the strong wind recorded by S01. Other stations also recorded strong winds around the same time, so I kept this reading instead of incorrectly treating a real storm as a sensor fault.

For S02, the pressure readings showed a consistent offset after a board replacement. The checker recommends correcting that offset.

## Least certain call: S07 wind

This is the network health report. It summarises the quality of each station's measurements and explains the main issues.

The most important recommendation is to visit S04, Foothills Community Garden. Its afternoon temperature readings increasingly differ from neighbouring stations, reaching approximately 4.9 degrees Celsius warmer by 11 April. The night-time readings remain much closer to the other stations.

The rain gauge also appears to have been blocked, so the affected rainfall readings should not be interpreted as normal dry weather.

For S06, the logger clock was six hours behind, and the report describes correcting the timestamps. For S07, wind-speed readings were excluded because the correct calibration factor could not be confirmed.

## Technician choice and close

I documented the reasoning behind the main decisions and the limitations of the analysis.

I used data validation, patterns over time, comparisons with neighbouring stations, GPS information, and maintenance records to investigate suspicious readings.

I also considered the risk of removing genuine weather events. Where the evidence did not support a reliable correction, I avoided guessing the corrected value.

Overall, this project demonstrates a reusable and explainable approach to weather-data quality checking. Thank you.
