# DECISIONS.md

## 1. My Approach

My main goal was to identify suspicious weather readings without treating every unusual value as a sensor fault. Weather conditions can change naturally, so I did not want to flag a reading only because it was different from the usual pattern.

I used a combination of data-quality rules, time-series analysis, station metadata, GPS information, and maintenance records. Based on the available evidence, I assigned one of three actions to each flagged issue:

- **Keep:** The reading appears valid, and there is not enough evidence to change or remove it.
- **Correct:** There is enough evidence to apply a specific correction, such as fixing a timestamp or a consistent measurement offset.
- **Exclude:** The readings are unreliable, and I could not confidently recover their correct values.

## 2. Main Issues I Identified

### S02 — Pressure Offset

I found that the pressure readings were approximately 6.6 hPa higher than expected after a board replacement. I treated this as a possible sensor offset rather than an actual change in atmospheric pressure.

I chose the correction approach because the shift appeared consistent. However, this correction should only be applied if the comparison with the surrounding stations and the maintenance information supports it.

### S03 — Frozen Temperature Readings

The temperature readings from S03 remained unchanged for an unusually long period, approximately 80 hours. I also found four readings containing error codes.

I considered these signs of a possible sensor or logging problem rather than normal weather behaviour. I chose to exclude the unreliable readings instead of assuming that the recorded temperature was correct.

### S04 — Temperature Drift and Rain Gauge Issue

I observed a daytime temperature difference of approximately 4.9°C compared with the expected pattern. I also found evidence that the rain gauge may have been blocked for around 4.5 days.

I considered these two issues separately because a temperature problem does not automatically explain a rainfall problem. I recommended a technician visit to investigate the station, particularly the temperature drift and the rain gauge.

### S05 — Humidity and GPS Metadata

The humidity readings remained above 100% for an extended period, which suggested a possible sensor issue.

I also found that the station's recorded GPS location did not match the expected location. The difference appeared to be caused by two swapped digits in the installation-sheet coordinates.

I treated the humidity problem as a data-quality issue and the location mismatch as a metadata issue. I would verify the original installation records before changing the station coordinates.

### S06 — Logger Clock Error

I found evidence that the S06 logger clock was approximately six hours behind GPS time following a reboot on 24 March.

I treated this as a timestamp problem rather than an actual weather problem. A time correction is appropriate only when the GPS evidence and the surrounding observations support the same time shift.

### S07 — Wind-Speed Change

After an anemometer replacement on 26 March, the wind-speed readings appeared to be approximately 0.27 times the usual level.

I considered a sensor installation or calibration problem as a possible explanation. I did not apply a correction factor because I could not confidently establish the correct factor from the available evidence. I therefore treated the affected readings as unreliable.

### S08 — Missing and Duplicated Records

I found approximately 36 hours of missing readings and a duplicated day's data on 5 April.

I treated the missing period as a data-coverage issue and the duplicated records as a possible logging or data-processing problem. I avoided treating duplicated measurements as independent observations because that could distort the analysis.

## 3. How I Decided What to Flag

I did not rely on a single threshold to identify all problems. I considered several types of evidence:

1. Whether readings were missing, duplicated, frozen, or outside plausible limits.
2. Whether a measurement changed suddenly or showed an unusual pattern over time.
3. Whether neighbouring stations showed a similar change.
4. Whether GPS information and station metadata were consistent.
5. Whether maintenance events could explain a change in the readings.

I used these checks to distinguish possible equipment or logging problems from genuine changes in weather. When the evidence was not strong enough to support a reliable correction, I preferred not to invent one.

## 4. Limitations and Uncertainty

The data alone cannot always prove why a sensor produced an unusual reading. For example, a sudden change could be caused by a faulty sensor, a maintenance event, or a real environmental change.

Some corrections require confirmation from calibration records, technician notes, or the original installation documentation. I would not treat a suspected cause as confirmed without that evidence.

I also recognise that unusual readings should not automatically be removed. Excluding valid extreme weather events could lead to an incorrect picture of the weather conditions.

## 5. Final Decision

My approach was to prioritise data reliability and explainability. I wanted every flag to have a clear reason and every correction to be supported by evidence.

Where the issue could be corrected confidently, I considered a correction. Where the data could not be trusted or recovered reliably, I preferred exclusion. Where the evidence did not justify either action, I preferred keeping the reading and avoiding unnecessary changes.

The final decisions should be reviewed alongside the generated `flags.csv`, the network health report, the station metadata, and the available maintenance evidence.