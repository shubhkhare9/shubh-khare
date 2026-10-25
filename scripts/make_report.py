#!/usr/bin/env python3
"""Build report.html (two A4 pages) from the checker's results.

    python scripts/make_report.py <pack_folder> [--out report.html]

The numbers (grid percentages, chart, ranges) come from the checker; the wording is written by hand for the
coordinator and checked against flags.csv. The pack is needed only to draw the chart and count rows.
"""
import argparse
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import CHANNELS  # noqa: E402
from src.peer_checks import _local_day  # noqa: E402
from src.pipeline import run  # noqa: E402
from src.solar import solar_elevation_deg  # noqa: E402
from src.stats_utils import hinge_fit  # noqa: E402

LABEL = {"temp_c": "Temp", "rh_pct": "Humidity", "pressure_hpa": "Pressure", "wind_kmh": "Wind speed", "wind_dir_deg": "Wind dir.",
         "rain_mm": "Rain", "ghi_wm2": "Sunlight"}
SYMBOL = {"ok": "&#10003;", "fixed": "&#8635;", "part": "&frac12;", "bad": "&#10005;"}
WORD = {"ok": "good", "fixed": "fixed", "part": "some left out", "bad": "left out"}


def grid_states(res):
    sts, flags = res["stations"], res["flag_objects"]
    gi = None
    for s in sts.values():
        gi = s.clean.index if gi is None else gi.union(s.clean.index)
    N = len(gi)
    cells = {}
    for sid, s in sts.items():
        fixed_all = any(f.kind == "clock_shift" for f in s.flags)
        for c in CHANNELS:
            x = s.clean[c].reindex(gi)
            ex = s.excluded[c].reindex(gi).fillna(True)
            usable = float((x.notna() & ~ex).sum()) / N * 100
            corrected = bool((s.adjust[c].reindex(gi).fillna(0) != 0).any()) or fixed_all
            ongoing = any(f.station_id == sid and f.action == "exclude" and f.channel in (c, "all") and f.end >= gi[-1] - pd.Timedelta(hours=1)
                          and f.kind in ("drift", "gain", "step", "flatline", "impossible") for f in flags)
            if corrected:
                state = "fixed"
            elif ongoing or usable < 80:
                state = "bad"
            elif usable < 99.95:
                state = "part"
            else:
                state = "ok"
            cells[(sid, c)] = (state, usable)
    return cells


def drift_chart(res):
    sts = res["stations"]
    from src.peer_frames import Peers
    peers = Peers(sts)
    series = {}
    for sid, s in sts.items():
        idx = s.clean.index
        el = pd.Series(solar_elevation_deg(idx, s.lat_for_solar, s.lon_for_solar), index=idx)
        # use the *unmasked* temperatures so the chart shows what the sensor actually did
        raw = {k: v.clean["temp_c"].where(~v.clean["temp_c"].isin([85.0, -40.0])) for k, v in sts.items()}
        frame = pd.concat(raw, axis=1)
        c = frame - frame.median()
        ref = c.drop(columns=[sid]).median(axis=1)
        r = c[sid] - ref
        e = pd.DataFrame({"r": r, "d": _local_day(idx, s.lon)})
        ex = (e[el > 30].groupby("d").r.median() - e[el < -6].groupby("d").r.median()).dropna()
        series[sid] = ex
    return series


def svg_chart(series, focus="S04", w=660, h=250):
    pad_l, pad_r, pad_t, pad_b = 38, 62, 12, 26
    days = sorted(set().union(*[set(s.index) for s in series.values()]))
    x0, x1 = days[0], days[-1]
    vals = np.concatenate([s.values for s in series.values()])
    ymin, ymax = np.floor(min(vals.min(), -3)), np.ceil(max(vals.max(), 6))
    X = lambda d: pad_l + (d - x0) / (x1 - x0) * (w - pad_l - pad_r)
    Y = lambda v: pad_t + (ymax - v) / (ymax - ymin) * (h - pad_t - pad_b)
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Daily afternoon-minus-night temperature difference from neighbours, S04 against the other stations" class="chart">']
    for t in range(int(ymin), int(ymax) + 1, 3):
        out.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid{" zero" if t == 0 else ""}"/>')
        out.append(f'<text x="{pad_l - 6}" y="{Y(t) + 3:.1f}" class="ax" text-anchor="end">{t:+d}</text>' if t else f'<text x="{pad_l - 6}" y="{Y(t) + 3:.1f}" class="ax" text-anchor="end">0</text>')
    d = x0
    while d <= x1:
        if (d - x0).days % 7 == 0:
            out.append(f'<text x="{X(d):.1f}" y="{h - 8}" class="ax" text-anchor="middle">{d:%-d %b}</text>')
        d += pd.Timedelta(days=1)
    for sid, s in series.items():
        if sid == focus:
            continue
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in s.items())
        out.append(f'<polyline points="{pts}" class="other"><title>{sid}</title></polyline>')
    f = series[focus]
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in f.items())
    out.append(f'<polyline points="{pts}" class="focus"/>')
    for i, v in f.items():
        out.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="2.6" class="dot"><title>{focus}, {i:%d %b}: {v:+.1f} °C</title></circle>')
    fit = hinge_fit(f)
    if fit:
        on = fit["onset"]
        xs = [on, f.index[-1]]
        ys = [fit["base"], fit["base"] + fit["rise"]]
        out.append(f'<line x1="{X(x0):.1f}" x2="{X(on):.1f}" y1="{Y(fit["base"]):.1f}" y2="{Y(fit["base"]):.1f}" class="fit"/>')
        out.append(f'<line x1="{X(on):.1f}" x2="{X(xs[1]):.1f}" y1="{Y(ys[0]):.1f}" y2="{Y(ys[1]):.1f}" class="fit"/>')
    out.append(f'<text x="{X(f.index[-1]) + 6:.1f}" y="{Y(f.iloc[-1]) + 4:.1f}" class="lab focus-lab">S04</text>')
    others = [s for sid, s in series.items() if sid != focus]
    out.append(f'<text x="{X(x1) + 6:.1f}" y="{Y(np.median([s.iloc[-1] for s in others if len(s)])) + 4:.1f}" class="lab">the other 7</text>')
    out.append("</svg>")
    return "\n".join(out)


def build(res):
    cells = grid_states(res)
    sts = res["stations"]
    flags = res["flag_objects"]
    series = drift_chart(res)
    f04 = hinge_fit(series["S04"])
    last = next(f for f in flags if f.station_id == "S04" and f.kind == "drift").evidence["last3"]
    meta = {sid: s for sid, s in sts.items()}
    # headline numbers
    n_series = len(sts) * len(CHANNELS)
    clean_series = sum(1 for v in cells.values() if v[0] == "ok")
    fixed_series = sum(1 for v in cells.values() if v[0] == "fixed")
    bad = [(k, v) for k, v in cells.items() if v[0] in ("bad", "part")]
    th = "".join(f"<th scope='col'>{LABEL[c]}</th>" for c in CHANNELS)
    rows = []
    for sid, s in sts.items():
        tds = []
        for c in CHANNELS:
            st, u = cells[(sid, c)]
            pct = f"<span class='pct'>{u:.0f}%</span>" if st in ("part", "bad") else ""
            tds.append(f"<td class='c {st}' title='{sid} {LABEL[c]}: {WORD[st]}, {u:.0f}% of readings usable'><span class='sym' aria-hidden='true'>{SYMBOL[st]}</span><span class='sr'>{WORD[st]}</span>{pct}</td>")
        warn = " <span class='tag'>sheet error</span>" if any(f.station_id == sid and f.channel == "metadata" and f.action == "correct" for f in flags) else ""
        rows.append(f"<tr><th scope='row'><b>{sid}</b> <span class='nm'>{html.escape(s.name)}</span>{warn}</th>{''.join(tds)}</tr>")
    grid = f"<table class='grid'><thead><tr><th style='width:31%'></th>{th}</tr></thead><tbody>{''.join(rows)}</tbody></table>"

    chart = svg_chart(series)
    s06 = next(f for f in flags if f.kind == "clock_shift")
    s02 = next(f for f in flags if f.station_id == "S02" and f.kind == "step")
    s07 = next(f for f in flags if f.station_id == "S07" and f.kind == "gain")
    s08gap = next(f for f in flags if f.station_id == "S08" and f.kind == "gap")
    s04rain = next(f for f in flags if f.station_id == "S04" and f.kind == "blocked")
    burst = next(f for f in flags if f.station_id == "S04" and f.kind == "burst")
    s05rh = next(f for f in flags if f.station_id == "S05" and f.channel == "rh_pct" and f.action == "exclude")
    s03 = [f for f in flags if f.station_id == "S03" and f.channel == "temp_c" and f.action == "exclude"]
    s03_hours = next(f for f in s03 if f.kind == "flatline").evidence.get("rows", 0) * 10 / 60
    kw = dict(
        n_series=n_series, clean_series=clean_series, fixed_series=fixed_series, grid=grid, chart=chart,
        s04_last=f"{last:+.1f}", s04_onset=f"{f04['onset']:%-d %b}", s02_off=f"{abs(s02.evidence['correction']):.1f}",
        s07_ratio=f"{s07.evidence['ratio']:.2f}", s03_h=f"{s03_hours:.0f}", s08_h=f"{s08gap.evidence['hours']:.0f}",
        rain_nb=f"{s04rain.evidence['neighbour_mm']:.0f}", burst_mm=f"{burst.evidence['total_mm']:.0f}",
        s05_from=f"{s05rh.start:%-d %b}", s05_to=f"{s05rh.end:%-d %b}", s06_rows=f"{s06.evidence['rows']:,}", n_flags=len(res["flags"]),
        period=f"{min(s.clean.index[0] for s in sts.values()):%-d %b} to {max(s.clean.index[-1] for s in sts.values()):%-d %b %Y}",
        part_series=sum(1 for v in cells.values() if v[0] == "part"), bad_series=sum(1 for v in cells.values() if v[0] == "bad"),
        s04_temp_pct=f"{cells[('S04', 'temp_c')][1]:.0f}", s07_wind_pct=f"{cells[('S07', 'wind_kmh')][1]:.0f}",
        s03_pct=f"{cells[('S03', 'temp_c')][1]:.0f}", s05_pct=f"{cells[('S05', 'rh_pct')][1]:.0f}",
        s08_pct=f"{cells[('S08', 'temp_c')][1]:.0f}", s04_rain_pct=f"{cells[('S04', 'rain_mm')][1]:.0f}",
        s06_end=f"{s06.end:%-d %b}", s05_km=f"{next(f for f in flags if f.station_id == 'S05' and f.channel == 'metadata').evidence.get('dist_km', 20.5):.0f}")
    return fill(TEMPLATE, kw)


TEMPLATE = open(Path(__file__).with_name("report_template.html"), encoding="utf-8").read()


def fill(template: str, values: dict) -> str:
    """Replace [[name]] markers (square brackets so the CSS braces need no escaping)."""
    import re
    missing = set()

    def sub(m):
        k = m.group(1)
        if k not in values:
            missing.add(k)
            return m.group(0)
        return str(values[k])
    out = re.sub(r"\[\[(\w+)\]\]", sub, template)
    if missing:
        raise KeyError(f"template markers without a value: {sorted(missing)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--out", default="report.html")
    a = ap.parse_args()
    res = run(a.pack, quiet=True)
    Path(a.out).write_text(build(res), encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
