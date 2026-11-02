"""Build one row per driver per race, using only information available before lights out."""
import numpy as np, pandas as pd
from pathlib import Path

FINISHED = lambda s: s == "Finished" or s == "Lapped" or s.startswith("+")
INCIDENT = {"Collision", "Accident", "Collision damage", "Spun off", "Damage", "Puncture"}

def classify(status):
    if FINISHED(status): return "finished"
    if status in INCIDENT: return "incident"
    if status == "Upcoming": return "upcoming"
    if status in {"Did not start", "Withdrew", "Did not qualify", "Not classified"}: return "dns"
    if status == "Disqualified": return "dsq"
    return "mechanical_or_retired"

def build(path="data", upcoming=None, cutoff=None):
    """upcoming: optional DataFrame of rows for races not yet run (results columns, position NaN).
    cutoff: race_id (season*100+round); drop every real result after it, to replay a past point in time."""
    r = pd.read_csv(f"{path}/results.csv")
    q = pd.read_csv(f"{path}/qualifying.csv")
    if cutoff is not None:
        r = r[r.season * 100 + r["round"] <= cutoff]
        q = q[q.season * 100 + q["round"] <= cutoff]
    if upcoming is not None:
        key = ["season", "round"]
        r = r.merge(upcoming[key].drop_duplicates(), on=key, how="left", indicator=True)
        r = pd.concat([r[r._merge == "left_only"].drop(columns="_merge"), upcoming[r.columns.drop("_merge")]])
        q = q.merge(upcoming[key].drop_duplicates(), on=key, how="left", indicator=True)
        q = pd.concat([q[q._merge == "left_only"].drop(columns="_merge"),
                       upcoming[["season", "round", "driver", "quali_pos", "q1", "q2", "q3"]]])
    df = r.merge(q, on=["season", "round", "driver"], how="left")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "position", "grid"]).reset_index(drop=True)
    df["race_id"] = df["season"] * 100 + df["round"]
    df["outcome"] = df["status"].map(classify)
    df = df[df["outcome"] != "dns"].copy()

    n_cars = df.groupby("race_id")["driver"].transform("count")
    df["field_size"] = n_cars
    # Pit lane starts are recorded as grid 0; treat them as starting last.
    df["grid_f"] = np.where(df["grid"] == 0, n_cars, df["grid"])

    # Qualifying pace: percentage gap to the fastest time, in Q1 (everyone runs it) and best overall.
    df["best_q"] = df[["q1", "q2", "q3"]].min(axis=1)
    for col, src in [("q1_gap", "q1"), ("best_gap", "best_q")]:
        fastest = df.groupby("race_id")[src].transform("min")
        df[col] = ((df[src] / fastest - 1) * 100).clip(upper=7)
    tm_best = df.groupby(["race_id", "team"])["best_gap"].transform("mean")
    tm_cnt = df.groupby(["race_id", "team"])["best_gap"].transform("count")
    df["teammate_gap"] = np.where(tm_cnt == 2, (df["best_gap"] - tm_best) * 2, np.nan)

    df["gained"] = df["grid_f"] - df["position"]
    df["dnf_mech"] = (df["outcome"] == "mechanical_or_retired").astype(float)
    df["dnf_incident"] = (df["outcome"] == "incident").astype(float)
    df.loc[df["outcome"] == "upcoming", ["dnf_mech", "dnf_incident"]] = np.nan

    # Rolling form, shifted by one race so a row never sees its own result.
    def roll(group_cols, col, window, name):
        df[name] = df.groupby(group_cols)[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean())

    roll(["driver"], "position", 5, "drv_pos_5")
    roll(["driver"], "gained", 10, "drv_gained_10")
    roll(["driver"], "dnf_incident", 20, "drv_incident_20")
    roll(["driver"], "best_gap", 5, "drv_qgap_5")
    df["drv_starts"] = df.groupby("driver").cumcount()

    # Team form: average per race first, then roll over races.
    team_race = df.groupby(["team", "race_id"]).agg(
        t_pos=("position", "mean"), t_gap=("best_gap", "mean"), t_mech=("dnf_mech", "mean")
    ).reset_index().sort_values("race_id")
    for col, w in [("t_pos", 5), ("t_gap", 5), ("t_mech", 10)]:
        team_race[f"team_{col[2:]}_{w}"] = team_race.groupby("team")[col].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean())
    df = df.merge(team_race.drop(columns=["t_pos", "t_gap", "t_mech"]), on=["team", "race_id"], how="left")

    # Circuit: how much running order usually changes there, from earlier seasons only.
    df["abs_change"] = (df["grid_f"] - df["position"]).abs()
    circ = df.groupby(["circuit", "season"])["abs_change"].mean().reset_index()
    circ["circ_shuffle"] = circ.groupby("circuit")["abs_change"].transform(
        lambda s: s.shift(1).expanding().mean())
    df = df.merge(circ[["circuit", "season", "circ_shuffle"]], on=["circuit", "season"], how="left")

    # Circuit retirement rate from earlier seasons (some tracks punish mistakes far more than others).
    df["not_finished"] = (~df["outcome"].isin(["finished", "upcoming"])).astype(float)
    df.loc[df["outcome"] == "upcoming", "not_finished"] = np.nan
    cd = df.groupby(["circuit", "season"])["not_finished"].mean().reset_index()
    cd["circ_dnf"] = cd.groupby("circuit")["not_finished"].transform(lambda s: s.shift(1).expanding().mean())
    df = df.merge(cd[["circuit", "season", "circ_dnf"]], on=["circuit", "season"], how="left")

    # Driver history at this circuit, earlier visits only.
    df = df.sort_values("date")
    df["drv_circ_pos"] = df.groupby(["driver", "circuit"])["position"].transform(
        lambda s: s.shift(1).expanding().mean())

    # Grid penalties and this weekend's car ranking.
    df["grid_penalty"] = (df["grid_f"] - df["quali_pos"]).clip(lower=0)
    team_best = df.groupby(["race_id", "team"])["best_gap"].transform("min")
    df["team_q_rank"] = team_best.groupby(df["race_id"]).rank(method="dense")

    # Sprint result this weekend (sprints run before the Grand Prix).
    if Path(f"{path}/sprint.csv").exists():
        sp = pd.read_csv(f"{path}/sprint.csv")
        if cutoff is not None:
            sp = sp[sp.season * 100 + sp["round"] <= cutoff]
        df = df.merge(sp[["season", "round", "driver", "sprint_pos", "sprint_grid"]],
                      on=["season", "round", "driver"], how="left")
        df["sprint_gained"] = df["sprint_grid"] - df["sprint_pos"]
        df = df.drop(columns="sprint_grid")

    # OpenF1 pace data (2023 onward): this weekend's practice and sprint, earlier races' race pace.
    if Path(f"{path}/of_laps.csv").exists():
        from pace import build_pace
        pace = build_pace(r.drop_duplicates(["season", "round", "driver"]), path)
        df = df.merge(pace, on=["season", "round", "driver"], how="left")
        df = df.sort_values(["date", "position", "grid"]).reset_index(drop=True)
        team_lr = df.groupby(["race_id", "team"])["lr_pace"].transform("mean")
        df["team_lr_pace"] = team_lr
        df["team_top_speed"] = df.groupby(["race_id", "team"])["top_speed"].transform("mean")
        df["team_sprint_pace"] = df.groupby(["race_id", "team"])["sprint_pace"].transform("min")
        # Rolling race pace from earlier races only. race_pace itself is this race's outcome, so it is
        # never used as a feature directly.
        roll(["driver"], "race_pace", 3, "drv_rpace_3")
        tr = df.groupby(["team", "race_id"])["race_pace"].min().reset_index().sort_values("race_id")
        tr["team_rpace_3"] = tr.groupby("team")["race_pace"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        df = df.merge(tr[["team", "race_id", "team_rpace_3"]], on=["team", "race_id"], how="left")

    if Path(f"{path}/weather.csv").exists():
        w = pd.read_csv(f"{path}/weather.csv")[["season", "round", "rain_mm", "rain_prob", "wet"]]
        df = df.merge(w, on=["season", "round"], how="left")

    df["era_2022"] = (df["season"] >= 2022).astype(int)
    df["era_2026"] = (df["season"] >= 2026).astype(int)
    df["grid_pct"] = df["grid_f"] / df["field_size"]
    return df.sort_values(["date", "position", "grid"]).reset_index(drop=True)

FEATURES = ["grid_f", "grid_pct", "field_size", "q1_gap", "best_gap", "teammate_gap",
            "drv_pos_5", "drv_gained_10", "drv_incident_20", "drv_qgap_5", "drv_starts",
            "team_pos_5", "team_gap_5", "team_mech_10", "circ_shuffle", "drv_circ_pos",
            "round", "era_2022", "era_2026"]

PACE_FEATURES = ["grid_penalty", "team_q_rank", "sprint_pos", "sprint_gained",
                 "lr_pace", "deg", "fp_best_gap", "top_speed", "sprint_pace",
                 "team_lr_pace", "team_top_speed", "team_sprint_pace", "drv_rpace_3", "team_rpace_3"]

if __name__ == "__main__":
    d = build()
    print(d.shape); print(d[FEATURES].describe().T[["count", "mean"]])
    d.to_csv("data/features.csv", index=False)
