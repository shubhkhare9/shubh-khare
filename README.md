Shubh Khare
shubhkhare.sk11@gmail.com
Video: [3-minute screen recording](https://drive.google.com/file/d/1zfcDi6UL5607gTeyJ1TV6sjUw6x_jvZ_/view?usp=sharing)
Run: python3 checker.py "/path/to/ridgeline_data_pack"

---

# Ridgeline weather-network checker

Reads one data pack (`data/*.csv`, `gps/*.nmea`, `stations.csv`, `maintenance_log.md`) and writes `flags.csv` into
the folder you run it from, with one row per real problem: `station_id, channel, start_utc, end_utc, issue, action, note`.
`action` is `exclude`, `correct` or `keep` (`keep` = looks odd but is real).

```
pip install -r requirements.txt          # numpy, pandas (scikit-learn and pytest are optional)
python checker.py ./ridgeline_data_pack  # writes ./flags.csv   (add -q for silence)
python -m pytest -q                      # 18 tests on synthetic packs; the real pack is never needed
python scripts/make_report.py ./ridgeline_data_pack --out report.html   # rebuilds the 2-page report
```

The data pack belongs to Aeron and is **not** in this repository (`.gitignore` blocks it). Nothing is hard-coded to
the eight stations, to March/April 2025, or to the maintenance log: stations are discovered from `data/*.csv`,
the log only adds a quotation to a flag that the data already produced. I checked this by running the checker on a copy
of the pack with the stations renamed, every date moved by exactly one year, and an empty log: the 23 flags were identical.

## What the checker decides, in order

The order matters. Time is verified before values, because a value stamped at the wrong hour is not a bad value.

| # | Check | Rule (and why this threshold) | Typical action |
|---|---|---|---|
| 1 | **Unreadable timestamps** | Row whose time cannot be parsed cannot be placed | exclude |
| 2 | **Replayed block** | Time steps backwards and every channel of the block equals rows already in the file (S08, 5 Apr) | exclude the copy, keep the first |
| 3 | **Wrong logger clock** | Time steps backwards but the block is new data. Shift is measured two independent ways: GPS time minus logger time over the block (median of valid, checksum-OK `$GPRMC` fixes) and "what shift makes the block continue the previous row". Agreement within one interval and no collisions after shifting, else exclude. The sun's daylight curve is quoted as a third witness | correct (shift), else exclude |
| 4 | **Duplicates, off-grid stamps, gaps** | Same time twice: identical copy dropped, conflicting copies excluded. A gap says whether the GPS file is also silent (logger down) or not (link lost). No interpolation | exclude |
| 5 | **GPS vs install sheet** | Median valid fix > 250 m from `stations.csv` (sheet has 4 decimals ≈ 11 m; fixes scatter ≈ 15 m). Corrected to GPS only if 90 % of fixes sit within 100 m of their median **and** GPS altitude agrees with the sheet elevation within 100 m | correct, else exclude |
| 6 | **Physical limits** | Temperature outside −90..60 °C, humidity > 100.5 %, wind < 0 or > 250 km/h, rain > 40 mm/10 min, pressure > 75 hPa from the standard atmosphere at the station's elevation, sunlight above the extra-terrestrial ceiling or > 30 W/m² with the sun 6° below the horizon (healthy stations stay under 11) | exclude |
| 7 | **Spikes** | Value more than max(floor, 8 × local robust scale) from the median of its 7 neighbours (floors: 5 °C, 30 % RH, 3 hPa, 35 km/h). Known error values (85, −40 …) are named. If ≥ 30 % of the other stations make the same jump at the same time it is weather | exclude, or keep |
| 8 | **Frozen readings** | N identical readings in a row (12 for temperature/humidity/direction/wind/sunlight, 24 for pressure; longest natural run in the healthy data is 6–9) **and** the neighbours moved over the same time. Humidity pinned at 100 % is kept when the neighbours' median is ≥ 95 % for 70 % of the run | exclude, or keep |
| 9 | **Level steps vs neighbours** | Hourly (station − median of others, each centred on its own median); abrupt shift ≥ 3 hPa / 15 % RH / 4 °C that lasts ≥ 12 h. For pressure the step is measured against the neighbours' own change in the same hour and corrected only if it agrees with the whole-segment shift and the level then stays constant | correct (pressure), else exclude |
| 10 | **Daytime-only temperature drift** | Per day: midday residual − night residual vs neighbours. A "flat, then ramp" fit must explain ≥ 40 % of the variance and rise ≥ 3 °C (healthy stations stay below 20 %). Only daytime rows (sun > 5°) are excluded | exclude |
| 11 | **Gain errors** | Wind and sunlight: daily means vs neighbours' median; ratio changes by ≥ 2× (wind) or 1.5× (sunlight) for ≥ 2 days. Not corrected, because the factor cannot be known | exclude |
| 12 | **Blocked rain gauge** | Gauge at exactly 0 for whole half-days in which most neighbours caught rain (≥ 10 mm in total). Then a burst ≥ 5 mm/interval right after is stored water | exclude |
| 13 | **Real extremes** | Each station's strongest wind episode, if ≥ 45 km/h: when most neighbours are in their own top 10 % within 3 h it is a storm, not a fault (otherwise kept with low confidence) | keep |
| 14 | **Volunteer log** | A note on the same station, about the same part, within ~1 day before/2 after is quoted in the flag. It never creates or removes a flag. A single bad sample is not attributed to a visit days away | evidence only |

Cross-station checks (9–13) need at least 4 stations; with fewer they are skipped and the run says so.

Optional, off by default: `python checker.py PACK --iforest` writes `iforest_candidates.csv`, an Isolation Forest ranking of
stretches the rules did not flag. It is a review aid, never a source of flags (there are no fault labels to train on, and
an unusual combination is as often weather as a fault).

## Layout

```
checker.py                 one-command entry point
src/data_loader.py         CSV, NMEA (with checksums), stations.csv, maintenance log
src/integrity_checks.py    clock, replays, duplicates, gaps
src/gps_checks.py          position, altitude, receiver health
src/weather_checks.py      physical limits and sun-position checks
src/time_series_checks.py  spikes, frozen readings, real extremes
src/peer_checks.py         level steps, daytime drift, gain, rain gauge
src/maintenance_checks.py  log as evidence
src/flag_builder.py        merge, apply, write
src/solar.py               sun position (used for the by-hand question too)
scripts/make_report.py     builds report.html from the checker's results
tests/                     synthetic packs with injected faults
DECISIONS.md  report.html  flags.csv  ai_logs/  docs/video_script.md
```

## Result on the supplied pack

23 flags: 12 exclude, 3 correct, 8 keep. The ones that matter: S06 clock 6 h slow after the 24 Mar reboot (corrected);
S04 daytime temperature drifting up to +4.9 °C and a blocked rain gauge; S07 wind reading 0.27× after the anemometer swap;
S03 temperature frozen for 80 h plus four error codes; S05 humidity stuck above 100 % and a latitude typo on the install
sheet; S08 36 h gap and a repeated day; S02 +6.6 hPa pressure step (corrected, after checking that the offset is
constant: slope 1.00 before and 1.01 after the swap).

## What I did not get to / known limits

- Thresholds were set looking at this one pack and then tested on synthetic packs and a renamed, re-dated copy. They are
  justified by physics or by the healthy part of the data, but a pack from another climate could need new numbers
  (every one is a named constant at the top of its module).
- Only **pressure**, **clock** and the **install sheet** are ever `correct`ed. Every other fault is `exclude`d, even where a
  factor looks guessable (S07), because a wrong correction is worse than a hole.
- Temperature and humidity steps must be abrupt to be flagged; slow drifts are caught only for the daytime-temperature pattern.
- The neighbour median assumes most stations are healthy. With 4-5 stations two simultaneous faults could mask each other.
- Day grouping uses solar time from longitude; a station with a wrong install-sheet longitude would misplace its days (the GPS check should flag it first).
- Local time in the volunteers' log is not converted to UTC; log notes are matched to flags with a window of a few days.
- Wind direction gets only range, frozen-vane and 'calm' logic; there is no neighbour comparison for it (direction is too local).
- `flags.csv` times for a re-timed block (S06) are given on the corrected UTC clock; the note gives the logger's own labels.
- I have not run it on stations with a logging interval other than 10 minutes beyond the synthetic test.
