# DECISIONS

*Draft in my voice. I will re-read it, rewrite it in my own words and be ready to talk it through. Numbers come from flags.csv.*

## 1. The visit: S04, Foothills Community Garden

Go around midday: the fault only shows with the sun up. **First:** before touching anything, stand a hand thermometer in the shade beside the shield for 10 minutes, and write down its reading, the logger's reading and the time. Repeat after half an hour. On 11 April the logger ran about 4.9 °C warmer than its neighbours in the afternoon, so a gap of several degrees confirms it is real and still growing. **Then** look for what changed on 15 March when the post was repainted: paint on the shield louvres, a bright white post or fence throwing sun into the shield, a shield that got moved. **Last 15 minutes:** pour a measured cup of water through the rain gauge to see it counts correctly after the 2 April clean-out, photograph everything.

**Why not S07 (second best).** Its wind has been lost since 26 March but the loss is not growing, and the likely cause (the new anemometer reporting m/s) can be settled by a photo of its label, with no visit. S04 damages temperature, the number a climate summary is about, it worsens each sunny week, and only someone standing there can find the cause.

## 2. Least certain: S07 wind, `exclude` rather than `correct`

*For a fault:* from 26 Mar 15:40, the day of the anemometer note, S07's wind drops to 0.27× its usual share of the neighbours' and stays there across wind bands; direction is normal. *For ×3.6:* 0.27 is within 15 % of 1/3.6 = 0.278, what a sensor reporting m/s into a km/h column would give. *Against:* neighbours are 40-120 km away, S07's own ratio wandered between about 0.8 and 1.2 before the swap, a different brand could simply have another calibration or start-up threshold, and I noticed the 3.6 match only after seeing the number. If I corrected and was wrong I would invent 16 days of wind; by excluding I only lose them. **What settles it:** the output unit on the new anemometer's label, or ten minutes beside a handheld anemometer.

## 3. Where the AI was wrong

- My first AI chat's plan said *"S06's data ends earlier than the other stations, while S08 has duplicate timestamps and a maintenance note about a website outage. We need to distinguish logging problems from website display problems."* Plausible, but S06's file is complete (6,048 rows): after the 24 Mar reboot its clock ran 6 h slow, so time steps back from 16:00 to 10:10. I noticed because file order and time order disagreed while the values carried on smoothly, and 435 GPS fixes read +6.00 h. S08 was not the website either: the GPS file is silent for the same 35.7 h, so the logger was off. That chat also listed S03's -40/85 readings but missed that its temperature sat at exactly 6.5 °C for 80 h; I found it by listing the longest runs of identical values.
- Claude's first night-time sunlight rule (over 10 W/m² with the sun 2° below the horizon) looked sensible but flagged about 40-50 readings at *every* healthy station, since the data are smoothed around twilight. Plotting the maximum reading against sun height for healthy stations (never above 11 W/m² below -6°) moved the rule to 30 W/m² at -6°.

## 4. Next year: 500 stations, nobody reads every flag

**Change:** compare each station with its 5-10 nearest neighbours at similar height, not everyone; run daily with state, so a fault is one open event with a start date and a running count, ranked by data lost and by whether it is still happening. Safe, reversible actions (error codes, replayed blocks, exact duplicates) apply themselves. **Drop:** the Isolation Forest, the "keep" flags for strong wind (they would bury readers at scale) and keyword-matching of volunteer notes (use a form instead). **Refuse to automate:** any `correct` that rewrites values or clocks (the pressure offset, +6 h) without a person or a reference reading; changing a station's location; dropping a station from a published summary; sending a technician; loosening thresholds because the output looks better.

## 5. By hand: when is the sun highest at S04 on 20 March 2025?

S04 is at 105.15° W. The Earth turns 15° an hour, so 1° of longitude = 4 minutes: 105.15 × 4 = 420.6 min = 7 h 00.6 min behind Greenwich. The sun is highest at 12:00 *sun* time. Near the equinox the sun runs about 7½ minutes slow (equation of time ≈ -7½ min), so it crosses the meridian 7½ min after 12:00.
12:00 + 7 h 00.6 min + 7.5 min ≈ **19:08 UTC** (13:08 local daylight time).
**Use:** the sunlight curve should peak around this time and be roughly symmetric. A wrong logger clock shows as a peak at the wrong hour: S06's after the reboot peaked 6 h early, which confirmed the clock fault without GPS.
