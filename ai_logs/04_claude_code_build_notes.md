# Build session notes (Claude), in order, dead ends included

1. Read the PDF brief and the three plan texts. Unzipped the pack and audited before writing code: ranges, intervals, duplicates, GPS vs logger time, GPS vs sheet.
2. Found early: S06 timestamps step back 5.8 h (not "data ends early"); S08 36 h gap + an exact repeated day; S05 GPS 20 km from sheet; S03 four error codes.
3. Dead end: a plain "10-minute jump > X" spike rule flagged S01's real 26 km/h gusts. Fixed by comparing with the median of 7 neighbours and asking the other stations.
4. Dead end: night-sunlight rule at 2 degrees below horizon / 10 W/m2 flagged ~40 rows at every healthy station. Looked at max GHI by sun height; moved to -6 deg / 30 W/m2.
5. Dead end: "any run of 12 identical values is stuck" would have flagged 67 identical night-time zero GHI readings and calm-wind zeros. Zeros are skipped; neighbours must have moved.
6. Hidden fault not in the plan or the log: S03 temperature frozen at 6.5 C for 481 readings, found by listing longest identical runs.
7. S04: daytime-only warm drift (found by comparing midday-minus-night residual with the neighbours); onset fitted as a ramp, not a step.
8. S02: tested that the pressure offset is constant (slope vs neighbours 1.00 before, 1.01 after) before allowing `correct`.
9. S07: gain 0.27 vs neighbours; close to 1/3.6, kept as exclude because unconfirmed.
10. While finishing, a half-rewritten version of the checker overwrote the committed one; the committed version was clearly better (corroborated "keep" flags, ramp fit for S04, merged S05 humidity run), so it was restored from git.
11. Unseen-pack test: renamed stations, shifted dates by one year, emptied the log: identical 23 flags. (A first test shifted by 200 days and raised "sunlight at night" everywhere: correct behaviour, because the sunlight values no longer matched the sun for the new season.)
12. Report built as two A4 pages and rendered to PDF to check page count.
