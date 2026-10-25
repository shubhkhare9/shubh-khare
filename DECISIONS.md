# DECISIONS.md

## 1. Technician visit: S04

I would spend the one-hour visit at S04. First I would inspect the rain-gauge funnel, tipping mechanism and wiring, then compare a manual tip with the logger count. The log records leaves and a bird nest in the funnel; during that period S04 recorded zero while neighbours measured 31.0 mm, followed by a 33.4 mm burst after cleaning. I would then inspect the temperature sensor's radiation shield, siting and ventilation. Its daytime residual rose to +4.9 °C while night readings stayed within 0.4 °C, and the post was repainted white. I would compare it with a calibrated handheld thermometer in sun and shade. S03 is my second choice because its temperature sensor froze for 80.2 hours and produced four error codes, but S04 has two affected instruments and physical causes that a short visit can inspect directly.

## 2. Least certain decision: exclude S07 wind readings

From 26 March, S07's wind speed was about 0.27 of its previous share of the neighbours' daily median, after an anemometer replacement. This supports a sensor or installation change, and the log confirms the replacement. However, 0.27 is close to 1/3.6, the conversion between metres per second and kilometres per hour; a unit configuration error could allow a specific correction. Local exposure or wind direction could also affect comparisons. I excluded the readings for now rather than assume a correction. The instrument model, logger unit configuration and a comparison against a calibrated handheld anemometer would resolve this.

## 3. Where AI was wrong

During development, an AI-generated simple spike rule treated a large single-station wind jump as a fault. Comparing the event with the other stations showed a shared gust, so I changed the check to use peer evidence and retain corroborated storms. An early sunlight rule also flagged nighttime readings across healthy stations. Checking the observed values against solar elevation showed that its threshold was too strict; I adjusted it using the healthy-station data. These iterations taught me to validate suggested rules against the whole network and known normal periods before trusting their flags.

## 4. What I would change for 500 stations

I would keep station discovery automatic, validate each file against a schema, and run independent station checks in parallel. I would replace manual inspection and fixed neighbour lists with a metadata-based spatial index and robust peer groups, while recording evidence, rule version and confidence for every flag. I would remove routine review of every normal row and send ranked exceptions to a dashboard. Human review should remain for ambiguous weather-versus-fault calls, material corrections, and exclusion decisions with operational impact; automatic corrections should require independent evidence and an audit trail.

## 5. Solar-noon estimate for S04, 20 March 2025

S04 longitude is 105.146° west. The longitude time offset is 105.146 ÷ 15 ≈ 7 h 00 min west of UTC. Around 20 March, the equation of time is about −8 min (apparent solar time runs about 8 min behind mean solar time). So solar noon in UTC is approximately 12:00 + 7:00 + 0:08 = **19:08 UTC**. This is an estimate; longitude and equation-of-time rounding give minute-level precision. Near solar noon, sunlight should be near its daily peak if the sensor and sky conditions allow it. Comparing the recorded curve with this expected timing helps identify a clock offset or sunlight-sensor problem; clouds can lower the peak without indicating a fault.
