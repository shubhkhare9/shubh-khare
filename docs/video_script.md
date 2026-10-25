# 3-minute screen recording (voice, no slides)

**0:00-0:15** Say your name. "Aeron checker: python checker.py PACK writes flags.csv. I'll show my strongest flag and my least certain flag."

**0:15-1:35 Strongest: S06 clock (flag `Logger clock wrong after restart`).**
1. Open data/S06.csv at row 3408-3410 (2025-03-24 16:00 then 10:10): "time steps back 5 h 50 min, but pressure 813.4 -> 813.5, temperature 14.0 -> 14.6 carry on smoothly - so it's new data on a wrong clock, not a replay."
2. Open gps/S06.nmea: logger stamp 11:00:01 but the sentence says 170000: "GPS is 6 h ahead, 435 fixes, spread 2 s."
3. Show the maintenance log 24 Mar: battery, reboot. "The volunteer thought the clock looked right on the screen."
4. Show flags.csv row: corrected +6 h, and the sunlight curve peaking 6 h early. Mention the same trick caught S08: GPS silent during the gap, exact repeated day.

**1:35-2:40 Least certain: S07 wind.** Show flags.csv row; plot/open S07 wind vs neighbours: drops to 0.27x on 26 Mar 15:40. "0.27 is close to 1/3.6, which would be m/s in a km/h column - but neighbours are 40-120 km away and I only noticed it after seeing the number. So I excluded, not corrected. One fact settles it: the unit on the new anemometer's label."

**2:40-3:00** "Technician goes to S04: afternoon temperature +4.9 C and growing; S07 can be settled by a photo. Everything is in DECISIONS.md and report.html. Thanks."

Record with any free recorder; upload unlisted; paste the link into line 3 of README.md.
