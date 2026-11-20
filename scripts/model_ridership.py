"""First ridership model: baselines vs a Random Forest, with neighborhood holdout.

Predicts log(1 + mean weekday entries) of each subway station complex from the
features of the hexagon it is in (data/model/training.csv, inputs = rows with
role 'feature' in data/model/feature_columns.csv) plus the student counts from
data/features/students.csv, joined by hex_id.

Evaluation holds out whole neighborhoods (2020 NTAs, GroupKFold, repeated with 3
shuffles and averaged), because nearby stations look alike and a random split
would flatter the model. A stricter leave-one-borough-out run is reported too.
Models, each adding one thing:

  mean        predict the average station (floor any model must beat)
  people      linear fit on log(residents + jobs + students in the hexagon), 1 feature
  linear      ridge regression on all features (log-transformed, standardized)
  rf_no_stud  Random Forest on all features except the student counts
  rf          Random Forest on all features
and, if scripts/features_catchment.py has been run, the same described by what is
within an 800 m walk instead of the 500 m hexagon:
  people_walk, linear_walk, rf_walk   walk-area features (+ competition + borough)
  rf_both                             hexagon and walk-area features together

The Random Forest with the lowest held-out error is the final model. Then:
permutation importance (on held-out folds), the Second Avenue Subway check
(train without its three 2017 stations, predict them), and the final Random
Forest scored on every candidate hexagon, ranked by new riders =
predicted riders x share of the 800 m walk area not already near rail.

Writes data/model/results/:
  cv_metrics.csv          one row per model and CV scheme
  oof_predictions.csv     each station's held-out prediction from every model
  importance.csv          permutation importance of each feature (Random Forest)
  candidate_predictions.csv  predicted and new riders for every candidate hexagon, ranked
  summary.md              the numbers above in one page
Usage:   python scripts/model_ridership.py
"""

from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from features_common import DATA

MODEL = DATA / "model"
OUT = MODEL / "results"
SEED = 0
REPEATS = 3   # neighborhood CV shuffles
STUDENT_COLS = ["students", "students_k12_public", "students_k12_private", "students_college",
                "student_density", "k12_schools", "college_campuses"]
SAS = {"477": "72 St (Q)", "476": "86 St (Q)", "475": "96 St (Q)"}   # Second Avenue Subway, opened 2017-01-01


def load():
    cols = pd.read_csv(MODEL / "feature_columns.csv")
    feats = cols[cols.role == "feature"].column.tolist()
    train = pd.read_csv(MODEL / "training.csv", comment="#", dtype={"station_complex_id": str})
    cand = pd.read_csv(MODEL / "candidates.csv", comment="#")
    stud = pd.read_csv(DATA / "features" / "students.csv", comment="#")
    train = train.merge(stud, on="hex_id", how="left", validate="many_to_one")
    cand = cand.merge(stud, on="hex_id", how="left", validate="one_to_one")
    # What is within an 800 m walk (scripts/features_catchment.py), if it has been built.
    ws_feats = []
    if (MODEL / "catchment_stations.csv").exists():
        ws_s = pd.read_csv(MODEL / "catchment_stations.csv", comment="#", dtype={"station_complex_id": str})
        ws_c = pd.read_csv(MODEL / "catchment_candidates.csv", comment="#")
        ws_feats = [c for c in ws_s.columns if c.startswith("ws_") and c != "ws_snap_m"]
        train = train.merge(ws_s, on="station_complex_id", how="left", validate="one_to_one")
        cand = cand.merge(ws_c, on="hex_id", how="left", validate="one_to_one")
    for c in train.columns:   # pandas reads 0/1 dummies as int, keep everything numeric as float
        if c in feats or c in ws_feats:
            train[c] = train[c].astype(float)
    return train, cand, feats + STUDENT_COLS, ws_feats


STATION_COLS = ["other_stations_1km", "nearest_other_station_m"]   # competition, same for both feature sets


def models(feats, ws_feats):
    def log_linear():
        # all features are counts, shares or distances (>= 0); log1p tames the long tails
        return make_pipeline(SimpleImputer(strategy="median"),
                             FunctionTransformer(lambda x: np.log1p(np.clip(x, 0, None))),
                             StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))

    def people(cols):
        return make_pipeline(ColumnTransformer([("p", make_pipeline(
            SimpleImputer(strategy="median"),
            FunctionTransformer(lambda x: np.log1p(x.sum(axis=1, keepdims=True)))), cols)]), LinearRegression())

    def rf():
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=0.33, n_jobs=-1, random_state=SEED))

    no_stud = [f for f in feats if f not in STUDENT_COLS]
    specs = {
        "mean": (DummyRegressor(), feats),
        "people": (people(["population", "jobs", "students"]), feats),
        "linear": (log_linear(), feats),
        "rf_no_stud": (rf(), no_stud),
        "rf": (rf(), feats),
    }
    if ws_feats:
        boro = [f for f in feats if f.startswith("boro_")]
        walk = ws_feats + STATION_COLS + boro
        specs |= {
            "people_walk": (people(["ws_population", "ws_jobs", "ws_students"]), walk),
            "linear_walk": (log_linear(), walk),
            "rf_walk": (rf(), walk),
            "rf_both": (rf(), feats + ws_feats),
        }
    return specs


def metrics(y_log, p_log):
    y, p = np.expm1(y_log), np.expm1(p_log)
    return {
        "r2_log": r2_score(y_log, p_log),
        "mae_log": mean_absolute_error(y_log, p_log),
        "rmse_log": np.sqrt(mean_squared_error(y_log, p_log)),
        "mae_riders": mean_absolute_error(y, p),
        "rmse_riders": np.sqrt(mean_squared_error(y, p)),
        "median_abs_pct_err": np.median(np.abs(p - y) / y) * 100,
        "spearman": spearmanr(y, p).statistic,
    }


def cross_validate(train, specs, y, splitter, groups):
    oof = {name: np.full(len(train), np.nan) for name in specs}
    for tr, te in splitter.split(train, y, groups):
        for name, (est, cols) in specs.items():
            m = clone(est).fit(train.iloc[tr][cols], y[tr])
            oof[name][te] = m.predict(train.iloc[te][cols])
    return oof


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, cand, feats, ws_feats = load()
    y = train.log_weekday_riders.to_numpy()
    specs = models(feats, ws_feats)
    print(f"{len(train)} stations, {train.nta2020.nunique()} neighborhoods, {len(feats)} hexagon features "
          f"(incl. {len(STUDENT_COLS)} student columns), {len(ws_feats)} 800 m walk features")

    # ---- cross-validation: whole neighborhoods held out (5 folds, repeated with 3 shuffles so one
    # lucky split doesn't decide a comparison), then whole boroughs
    schemes = {
        **{f"neighborhood_5fold_s{s}": (GroupKFold(n_splits=5, shuffle=True, random_state=s), train.nta2020)
           for s in range(REPEATS)},
        "leave_one_borough_out": (LeaveOneGroupOut(), train.borough),
    }
    rows, oof_main = [], None
    for scheme, (splitter, groups) in schemes.items():
        oof = cross_validate(train, specs, y, splitter, groups)
        if scheme == "neighborhood_5fold_s0":
            oof_main = oof
        for name, p in oof.items():
            rows.append({"scheme": scheme, "model": name, **metrics(y, p)})
        print(f"  {scheme}: done")
    runs = pd.DataFrame(rows)
    runs["scheme"] = runs.scheme.str.replace(r"_s\d+$", "", regex=True)
    cv = runs.groupby(["scheme", "model"], sort=False).mean().reset_index()
    cv["r2_log_sd"] = runs.groupby(["scheme", "model"], sort=False).r2_log.std().values   # spread over shuffles
    cv.round(4).to_csv(OUT / "cv_metrics.csv", index=False)

    # The Random Forest variant with the lowest held-out-neighborhood error is the final model.
    nb_rf = cv[(cv.scheme == "neighborhood_5fold") & cv.model.str.startswith("rf")]
    best = nb_rf.sort_values("rmse_log").model.iloc[0]
    print(f"  best Random Forest: {best}")

    oof_df = train[["station_complex_id", "station_complex", "borough", "nta2020", "hex_id", "weekday_riders"]].copy()
    for name, p in oof_main.items():
        oof_df[f"pred_{name}"] = np.round(np.expm1(p))
    oof_df["best_error_pct"] = np.round((oof_df[f"pred_{best}"] - oof_df.weekday_riders)
                                        / oof_df.weekday_riders * 100, 1)
    oof_df.sort_values("best_error_pct", key=abs, ascending=False).to_csv(OUT / "oof_predictions.csv", index=False)

    # ---- permutation importance on held-out neighborhoods (best RF)
    est, cols = specs[best]
    imp = np.zeros(len(cols))
    splitter, groups = schemes["neighborhood_5fold_s0"]
    for tr, te in splitter.split(train, y, groups):
        m = clone(est).fit(train.iloc[tr][cols], y[tr])
        r = permutation_importance(m, train.iloc[te][cols], y[te], n_repeats=5, random_state=SEED,
                                   scoring="neg_root_mean_squared_error", n_jobs=-1)
        imp += r.importances_mean / splitter.n_splits
    importance = pd.DataFrame({"feature": cols, "rmse_increase_log": np.round(imp, 5)}) \
        .sort_values("rmse_increase_log", ascending=False)
    importance.to_csv(OUT / "importance.csv", index=False)

    # ---- Second Avenue Subway: train without its stations, predict them
    sas = train.station_complex_id.isin(SAS)
    m = clone(est).fit(train.loc[~sas, cols], y[~sas])
    sas_check = train.loc[sas, ["station_complex", "weekday_riders"]].assign(
        predicted=np.round(np.expm1(m.predict(train.loc[sas, cols]))))

    # ---- final model on every station, scored on the candidate hexagons
    final = clone(est).fit(train[cols], y)
    rf = final[-1]
    x_cand = final[:-1].transform(cand[cols])
    trees = np.stack([t.predict(x_cand) for t in rf.estimators_])
    cand_out = cand[["hex_id", "borough", "nta2020", "ntaname", "uncovered_share_subway", "uncovered_share_any_rail",
                     "nonresidential_nta", "share_unbuilt_unlivable", "population", "jobs", "students",
                     "walk_to_entrance_m", "near_any_proposal"]].copy()
    cand_out["predicted_riders"] = np.round(np.expm1(trees.mean(axis=0)))
    cand_out["predicted_p10"] = np.round(np.expm1(np.percentile(trees, 10, axis=0)))
    cand_out["predicted_p90"] = np.round(np.expm1(np.percentile(trees, 90, axis=0)))
    cand_out["new_riders"] = np.round(cand_out.predicted_riders * cand_out.uncovered_share_any_rail.fillna(0))
    cand_out["extrapolation"] = (cand_out.borough == "Staten Island").astype(int)
    cand_out["eligible"] = ((cand_out.nonresidential_nta != 1) & (cand_out.share_unbuilt_unlivable <= 0.5)
                            & (cand_out.extrapolation == 0)).astype(int)
    cand_out = cand_out.sort_values(["eligible", "new_riders"], ascending=False)
    cand_out["rank"] = np.where(cand_out.eligible == 1, np.arange(1, len(cand_out) + 1), np.nan)
    cand_out.to_csv(OUT / "candidate_predictions.csv", index=False)

    # ---- summary
    nb = cv[cv.scheme == "neighborhood_5fold"].set_index("model")
    lobo = cv[cv.scheme == "leave_one_borough_out"].set_index("model")
    top = cand_out[cand_out.eligible == 1].head(15)
    fmt = lambda df: df.round(3).to_markdown()
    lines = [
        f"# Ridership model results ({date.today().isoformat()})",
        "",
        f"{len(train)} subway station complexes, target log(1 + mean weekday entries, 2025). "
        f"Hexagon features: {len(feats)} (the station's 500 m hexagon); walk features: {len(ws_feats)} (what is "
        "within an 800 m walk, `_walk` models). Scores are on stations the model did not train on. "
        f"Final model (lowest held-out error among the Random Forests): **{best}**.",
        "",
        f"## Whole neighborhoods held out (5 folds by NTA, mean of {REPEATS} shuffles; r2_log_sd = spread)",
        "", fmt(nb.drop(columns="scheme")), "",
        "## Whole boroughs held out", "", fmt(lobo.drop(columns="scheme")), "",
        "`r2_log`, `mae_log`, `rmse_log` are on the log scale the model trains on; `mae_riders` is in weekday "
        "entries; `median_abs_pct_err` is the typical miss in percent; `spearman` is rank agreement "
        "(what matters for ranking sites).",
        "",
        "## Most important features (Random Forest, permutation on held-out neighborhoods)",
        "", fmt(importance.head(15).set_index("feature")), "",
        "## Second Avenue Subway check (trained without these three stations)",
        "", fmt(sas_check.set_index("station_complex")), "",
        "## Top 15 candidate hexagons by new riders",
        "",
        "Eligible = not a park/airport/cemetery neighborhood, at most half unbuildable land, not Staten Island "
        "(no training stations there). New riders = predicted riders x share of the 800 m walk area not "
        "already within 800 m of the subway, LIRR, Metro-North or PATH.",
        "", fmt(top[["hex_id", "borough", "ntaname", "predicted_riders", "predicted_p10", "predicted_p90",
                     "uncovered_share_any_rail", "new_riders", "near_any_proposal"]].set_index("hex_id")), "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nHeld-out neighborhoods:")
    print(nb[["r2_log", "mae_log", "rmse_log", "mae_riders", "median_abs_pct_err", "spearman"]].round(3).to_string())
    print("\nHeld-out boroughs:")
    print(lobo[["r2_log", "mae_riders", "median_abs_pct_err", "spearman"]].round(3).to_string())
    print("\nTop features:\n" + importance.head(10).to_string(index=False))
    print("\nSecond Avenue Subway:\n" + sas_check.to_string(index=False))
    print("\nTop candidates:\n" + top[["hex_id", "borough", "ntaname", "predicted_riders", "uncovered_share_any_rail",
                                       "new_riders"]].head(10).to_string(index=False))
    print(f"\nwrote {OUT.relative_to(DATA.parent)}/: cv_metrics.csv, oof_predictions.csv, importance.csv, "
          "candidate_predictions.csv, summary.md")


if __name__ == "__main__":
    main()
