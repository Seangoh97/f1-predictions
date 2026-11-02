"""Turn OpenF1 practice and race laps into pace features.

Practice (this weekend):  long-run pace, tyre degradation, one-lap pace, top speed
Sprint (this weekend):    clean-air sprint race pace
Race (earlier races):     clean race pace, used as rolling car and driver form
All pace numbers are % gap to the best car in that session, so 0 = fastest."""
import numpy as np, pandas as pd

def load(path="data"):
    ses = pd.read_csv(f"{path}/of_sessions.csv")
    laps = pd.read_csv(f"{path}/of_laps.csv")
    st = pd.read_csv(f"{path}/of_stints.csv")
    return ses, laps, st

def tag_stints(laps, st):
    """Attach compound, stint number and tyre age to each lap."""
    st = st.dropna(subset=["lap_start", "lap_end"])
    rows = []
    for r in st.itertuples():
        for lap in range(int(r.lap_start), int(r.lap_end) + 1):
            rows.append((r.session_key, r.driver_number, lap, r.stint_number, r.compound,
                         (r.tyre_age_at_start or 0) + lap - r.lap_start, lap == r.lap_end))
    tag = pd.DataFrame(rows, columns=["session_key", "driver_number", "lap_number", "stint",
                                      "compound", "tyre_age", "stint_last"])
    tag = tag.drop_duplicates(["session_key", "driver_number", "lap_number"])
    return laps.merge(tag, on=["session_key", "driver_number", "lap_number"], how="left")

def clean(laps):
    l = laps.dropna(subset=["lap_duration"])
    l = l[(l.is_pit_out_lap != True) & (l.stint_last != True) & (l.lap_number > 1)]
    return l

def practice_features(laps):
    """Long runs: a stint with 5+ laps within 4% of that stint's median lap."""
    l = clean(laps).dropna(subset=["stint"]).copy()
    l = l[l.compound.isin(["SOFT", "MEDIUM", "HARD"])]
    l = l[l.lap_duration < l.groupby("session_key").lap_duration.transform("min") * 1.12]
    g = l.groupby(["session_key", "driver_number", "stint"])["lap_duration"]
    l = l[l.lap_duration < g.transform("median") * 1.04].copy()
    l["n"] = l.groupby(["session_key", "driver_number", "stint"])["lap_duration"].transform("count")
    lr = l[l.n >= 5].copy()

    out = []
    for sk, s in lr.groupby("session_key"):
        if s.driver_number.nunique() < 8: continue
        # Long-run pace per compound: each driver's median long-run lap against the median of everyone
        # who ran that compound (needs 4+ drivers on it). Drivers on several compounds get the average.
        med = s.groupby(["compound", "driver_number"]).lap_duration.median().reset_index()
        med = med[med.groupby("compound").driver_number.transform("count") >= 4]
        if med.empty: continue
        med["rel"] = (med.lap_duration / med.groupby("compound").lap_duration.transform("median") - 1) * 100
        pace = med.groupby("driver_number").rel.mean()
        pace = (pace - pace.min()).clip(upper=5)
        drivers = list(pace.index)
        # Degradation: slope of lap time against tyre age within each long stint, per driver.
        deg = {}
        for (d, _), stint in s.groupby(["driver_number", "stint"]):
            if stint.tyre_age.nunique() >= 4:
                deg.setdefault(d, []).append(np.polyfit(stint.tyre_age, stint.lap_duration, 1)[0])
        deg = pd.Series({d: np.mean(v) for d, v in deg.items()})
        deg = deg - deg.median()
        out.append(pd.DataFrame({"session_key": sk, "driver_number": drivers,
                                 "lr_pace": pace.reindex(drivers).values, "deg": deg.reindex(drivers).values}))
    lrdf = pd.concat(out) if out else pd.DataFrame(columns=["session_key", "driver_number", "lr_pace", "deg"])

    # One-lap pace and top speed from all practice laps.
    best = laps.dropna(subset=["lap_duration"]).groupby(["session_key", "driver_number"]).agg(
        best=("lap_duration", "min"), vmax=("st_speed", lambda x: x.quantile(0.95))).reset_index()
    best["fp_best_gap"] = (best.best / best.groupby("session_key").best.transform("min") - 1) * 100
    best["fp_best_gap"] = best.fp_best_gap.clip(upper=5)
    best["top_speed"] = best.vmax - best.groupby("session_key").vmax.transform("median")
    return best[["session_key", "driver_number", "fp_best_gap", "top_speed"]].merge(
        lrdf, on=["session_key", "driver_number"], how="left")

def race_pace(laps):
    """Median clean race lap per driver, ignoring laps run under SC/VSC (slow for the whole field)."""
    l = clean(laps).copy()
    field = l.groupby(["session_key", "lap_number"]).lap_duration.median()
    ref = field.groupby("session_key").quantile(0.25)
    l = l.join(field.rename("field_lap"), on=["session_key", "lap_number"]).join(ref.rename("ref"), on="session_key")
    l = l[(l.field_lap < l.ref * 1.07) & (l.lap_duration < l.ref * 1.07)]
    p = l.groupby(["session_key", "driver_number"]).lap_duration.agg(["median", "count"]).reset_index()
    p = p[p["count"] >= 5]
    p["race_pace"] = (p["median"] / p.groupby("session_key")["median"].transform("min") - 1) * 100
    return p[["session_key", "driver_number", "race_pace"]]

def build_pace(results, path="data"):
    """Return one row per (season, round, driver) with pace features. Needs results with car numbers."""
    ses, laps, st = load(path)
    laps = tag_stints(laps, st)
    ses["date"] = pd.to_datetime(ses.date_start).dt.tz_localize(None)
    # Match each OpenF1 meeting to a Jolpica race by date (race within 4 days after the session).
    races = results[["season", "round", "date"]].drop_duplicates()
    races["date"] = pd.to_datetime(races.date)
    keys = []
    for r in ses.itertuples():
        d = (races.date - r.date.normalize()).dt.days
        hit = races[(d >= -1) & (d <= 4)]
        keys.append((hit.season.iloc[0], hit["round"].iloc[0]) if len(hit) else (np.nan, np.nan))
    ses[["season", "round"]] = pd.DataFrame(keys, index=ses.index)
    ses = ses.dropna(subset=["season"])

    prac_ids = ses[ses.session_name.str.startswith("Practice")].session_key
    fp = practice_features(laps[laps.session_key.isin(prac_ids)])
    fp = fp.merge(ses[["session_key", "season", "round"]], on="session_key")
    fp = fp.groupby(["season", "round", "driver_number"])[["lr_pace", "deg", "fp_best_gap", "top_speed"]].agg(
        {"lr_pace": "mean", "deg": "mean", "fp_best_gap": "min", "top_speed": "mean"}).reset_index()

    rp = race_pace(laps[laps.session_key.isin(ses[ses.session_name.isin(["Race", "Sprint"])].session_key)])
    rp = rp.merge(ses[["session_key", "season", "round", "session_name"]], on="session_key")
    gp = rp[rp.session_name == "Race"].drop(columns=["session_key", "session_name"])
    spr = rp[rp.session_name == "Sprint"].rename(columns={"race_pace": "sprint_pace"}).drop(
        columns=["session_key", "session_name"])

    out = fp.merge(gp, on=["season", "round", "driver_number"], how="outer").merge(
        spr, on=["season", "round", "driver_number"], how="outer")
    nums = results[["season", "round", "number", "driver"]].rename(columns={"number": "driver_number"})
    out = out.merge(nums, on=["season", "round", "driver_number"], how="inner")
    out[["season", "round"]] = out[["season", "round"]].astype(int)
    return out.drop(columns="driver_number")
