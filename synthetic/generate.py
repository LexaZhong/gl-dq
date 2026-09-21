"""Generate a synthetic gl_master table (plus source-of-truth tables) for testing gl_dq.

    python synthetic/generate.py --n-policies 20000 --seed 42 --out data/
    python synthetic/generate.py --no-inject --out data/clean/

Writes <out>/gl_synth.duckdb (tables gl_master_synth, sot_premium_synth, sot_loss_synth)
and one parquet file per table. SOT tables are built from the clean data *before*
the issues in injected_issues.yaml are injected.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

AS_OF = pd.Timestamp("2025-06-30")
EFF_START, EFF_END = pd.Timestamp("2018-01-01"), pd.Timestamp("2024-12-31")

# coverage -> (premium share vs premises, inclusion probability, mean severity)
COVERAGES = {
    "Premises/Operations": (1.00, 1.00, 15_000),
    "Products/Completed Ops": (0.60, 0.70, 25_000),
    "Personal & Advertising Injury": (0.10, 0.60, 20_000),
    "Medical Payments": (0.05, 0.50, 3_000),
    "Liquor Liability": (0.40, 0.08, 40_000),
}
# exposure base -> (median exposure units, base rate per unit, share of class codes)
EXPO_BASES = {
    "Sales": (800, 2.0, 0.35),
    "Payroll": (400, 4.0, 0.30),
    "Area": (5, 150.0, 0.15),
    "Units": (10, 90.0, 0.12),
    "Admissions": (20, 40.0, 0.08),
}
# state -> (weight, zip3 range, rate factor)
STATES = {
    "CA": (0.14, (900, 961), 1.25), "TX": (0.11, (750, 799), 1.05), "NY": (0.09, (100, 149), 1.35),
    "FL": (0.08, (320, 349), 1.15), "IL": (0.06, (600, 629), 1.10), "PA": (0.05, (150, 196), 1.00),
    "OH": (0.05, (430, 458), 0.90), "GA": (0.05, (300, 319), 0.95), "NC": (0.05, (270, 289), 0.90),
    "NJ": (0.05, (70, 89), 1.20), "WA": (0.04, (980, 994), 1.00), "AZ": (0.04, (850, 865), 0.95),
    "MA": (0.04, (10, 27), 1.10), "CO": (0.04, (800, 816), 0.95), "MI": (0.05, (480, 499), 1.00),
}
# source -> (policy share, target loss ratio, max locations, max items per location, max classes)
SOURCES = {
    "BOP": (0.50, 0.58, 3, 1, 1),
    "BMQ": (0.30, 0.63, 5, 3, 1),
    "CMQ": (0.20, 0.66, 1, 1, 3),
}
DEDUCTIBLES = np.array([0, 250, 500, 1_000, 2_500, 5_000])
DED_CREDIT = dict(zip(DEDUCTIBLES, [1.00, 0.98, 0.96, 0.93, 0.88, 0.83]))
N_CLASSES = 60
# ISO-style rating dimensions. Limit ladder: (each occurrence, general aggregate, share, rate factor);
# higher limits cost more, so the premium mix is not uniform across the ladder.
LIMITS = [
    (1_000_000, 2_000_000, 0.46, 1.00),
    (2_000_000, 4_000_000, 0.28, 1.18),
    (1_000_000, 1_000_000, 0.11, 0.94),
    (5_000_000, 5_000_000, 0.10, 1.42),
    (500_000, 1_000_000, 0.05, 0.86),
]
TERRITORIES_PER_STATE = 4  # trr_cd = <state index><territory>, e.g. 0103
# market segment -> (share, exposure multiplier)
MARKET_SEGMENTS = {"SMALL": (0.62, 0.55), "MIDDLE": (0.31, 1.7), "LARGE": (0.07, 6.0)}


def _class_table(rng: np.random.Generator) -> pd.DataFrame:
    codes = rng.choice(np.arange(10_010, 98_999), size=N_CLASSES, replace=False)
    bases = list(EXPO_BASES)
    base = rng.choice(bases, size=N_CLASSES, p=[EXPO_BASES[b][2] for b in bases])
    weights = 1 / np.arange(1, N_CLASSES + 1) ** 1.1
    return pd.DataFrame({
        "class1_cd": codes.astype(str),
        "expn_bs": base,
        "rate_mult": rng.lognormal(0, 0.4, N_CLASSES),
        "weight": weights / weights.sum(),
    })


def _group_index(counts: np.ndarray) -> np.ndarray:
    """1-based position within each repeated group."""
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    return np.arange(counts.sum()) - starts + 1


def generate_clean(n_policies: int, rng: np.random.Generator) -> pd.DataFrame:
    classes = _class_table(rng)
    n = n_policies

    # --- policies ---------------------------------------------------------
    src_names = list(SOURCES)
    src = rng.choice(src_names, size=n, p=[SOURCES[s][0] for s in src_names])
    eff = EFF_START + pd.to_timedelta(rng.integers(0, (EFF_END - EFF_START).days + 1, n), unit="D")
    term_days = np.where(rng.random(n) < 0.85, 365, 182)
    cancelled = rng.random(n) < 0.08
    cancel_days = (rng.random(n) * (term_days - 31) + 30).astype(int)
    exp = eff + pd.to_timedelta(np.where(cancelled, cancel_days, term_days), unit="D")
    stat = np.where(cancelled, "Cancelled", np.where(exp < AS_OF, "Expired", "Active"))
    unearned = np.where(cancelled, 1 - cancel_days / term_days, 0.0)
    state_names = list(STATES)
    w = np.array([STATES[s][0] for s in state_names])
    state = rng.choice(state_names, size=n, p=w / w.sum())
    primary_cls = rng.choice(N_CLASSES, size=n, p=classes["weight"].to_numpy())
    csl = rng.random(n) < 0.20
    bi_ded = np.where(csl, np.nan, rng.choice(DEDUCTIBLES, n))
    pd_ded = np.where(csl, np.nan, rng.choice(DEDUCTIBLES, n))
    csl_ded = np.where(csl, rng.choice(DEDUCTIBLES, n), np.nan)
    # --- ISO rating dimensions ---------------------------------------------
    # territory: a few per state, skewed so a handful of cells carry most of the book
    terr_idx = np.minimum(rng.geometric(0.55, n) - 1, TERRITORIES_PER_STATE - 1)
    state_idx = pd.Series(state).map({s: i for i, s in enumerate(state_names)}).to_numpy()
    trr_cd = [f"{si + 1:02d}{ti + 1:02d}" for si, ti in zip(state_idx, terr_idx)]
    lim_idx = rng.choice(len(LIMITS), size=n, p=[l[2] for l in LIMITS])
    each_occ = np.array([LIMITS[i][0] for i in lim_idx], dtype=float)
    genl_ag = np.array([LIMITS[i][1] for i in lim_idx], dtype=float)
    limit_factor = np.array([LIMITS[i][3] for i in lim_idx])
    seg_names = list(MARKET_SEGMENTS)
    mm_seg = rng.choice(seg_names, size=n, p=[MARKET_SEGMENTS[s][0] for s in seg_names])
    policies = pd.DataFrame({
        "src": src,
        "pol_num": [f"{s}{i:07d}" for s, i in zip(src, range(1, n + 1))],
        "pol_eff_dt": eff, "pol_exp_dt": exp, "pol_stat": stat,
        "term_frac": term_days / 365, "unearned": unearned,
        "base_tx": np.where(rng.random(n) < 0.45, "Renewal", "New Business"),
        "loc_st_abbr": state, "primary_cls": primary_cls, "trr_cd": trr_cd,
        "each_occ_lmt_amt": each_occ, "genl_ag_lmt_amt": genl_ag, "limit_factor": limit_factor,
        "mm_seg_cd": mm_seg, "seg_expo_mult": [MARKET_SEGMENTS[s][1] for s in mm_seg],
        "bi_ded_amt": bi_ded, "pd_ded_amt": pd_ded, "csl_ded_amt": csl_ded,
    })

    # --- units: BOP/BMQ = location x item, CMQ = class code -----------------
    max_loc = policies["src"].map({s: v[2] for s, v in SOURCES.items()}).to_numpy()
    max_cls = policies["src"].map({s: v[4] for s, v in SOURCES.items()}).to_numpy()
    n_units = np.where(max_cls > 1, rng.integers(1, max_cls + 1), rng.integers(1, max_loc + 1))
    units = policies.loc[np.repeat(policies.index, n_units)].reset_index(drop=True)
    pos = _group_index(n_units)
    is_cmq = units["src"].eq("CMQ").to_numpy()
    units["rsk_loc_id"] = pd.array(np.where(is_cmq, 0, pos), dtype="Int64")
    units.loc[is_cmq, "rsk_loc_id"] = pd.NA
    # CMQ: distinct class per unit; BOP/BMQ: mostly the primary class
    other_cls = rng.choice(N_CLASSES, size=len(units), p=classes["weight"].to_numpy())
    loc_cls = np.where(rng.random(len(units)) < 0.7, units["primary_cls"], other_cls)
    units["cls"] = np.where(is_cmq, (units["primary_cls"] + (pos - 1) * 7) % N_CLASSES, loc_cls)
    zr = units["loc_st_abbr"].map(lambda s: STATES[s][1])
    zip3 = [rng.integers(lo, hi + 1) for lo, hi in zr]
    units["loc_zipcd"] = [f"{z:03d}{rng.integers(0, 100):02d}" for z in zip3]

    n_items = np.where(units["src"].eq("BMQ"), rng.integers(1, SOURCES["BMQ"][3] + 1, len(units)), 1)
    items = units.loc[np.repeat(units.index, n_items)].reset_index(drop=True)
    items["rsk_itm_id"] = pd.array(_group_index(n_items), dtype="Int64")
    items.loc[~items["src"].eq("BMQ"), "rsk_itm_id"] = pd.NA
    items = items.merge(classes.drop(columns="weight"), left_on="cls", right_index=True, how="left")
    median = items["expn_bs"].map(lambda b: EXPO_BASES[b][0])
    scale = np.where(items["src"].eq("CMQ"), 2.5, 1.0)
    items["expo_unit"] = np.maximum(np.round(
        median * scale * items["seg_expo_mult"] * rng.lognormal(0, 0.9, len(items))
        * items["term_frac"], 2), 0.01)

    # --- coverage rows ----------------------------------------------------
    covg_names = list(COVERAGES)
    mask = rng.random((len(items), len(covg_names))) < np.array([COVERAGES[c][1] for c in covg_names])
    mask[:, 0] = True
    item_idx, covg_idx = np.nonzero(mask)
    rows = items.loc[item_idx].reset_index(drop=True)
    rows["covg_type_desc"] = np.array(covg_names)[covg_idx]
    rows["expo_amt"] = rows["expo_unit"]

    base_rate = rows["expn_bs"].map(lambda b: EXPO_BASES[b][1])
    state_factor = rows["loc_st_abbr"].map(lambda s: STATES[s][2])
    covg_share = rows["covg_type_desc"].map(lambda c: COVERAGES[c][0])
    ded = rows["csl_ded_amt"].fillna(rows["bi_ded_amt"]).astype(int)
    ded_credit = ded.map(DED_CREDIT)
    gross = (rows["expo_amt"] * base_rate * rows["rate_mult"] * state_factor * covg_share
             * ded_credit * rows["limit_factor"] * rng.lognormal(0, 0.25, len(rows)))
    rows["tx_type_nm"] = rows["base_tx"]
    # CMQ carries net premium on the base row; BOP/BMQ book a separate cancellation row
    rows["tot_wrtn_prm_amt"] = np.round(np.where(rows["src"].eq("CMQ"), gross * (1 - rows["unearned"]), gross), 2)
    rows["gross"] = gross

    # --- claims on base rows ----------------------------------------------
    lr = rows["src"].map(lambda s: SOURCES[s][1])
    mean_sev = rows["covg_type_desc"].map(lambda c: COVERAGES[c][2]).to_numpy()
    earned = rows["tot_wrtn_prm_amt"] * np.where(rows["src"].eq("CMQ"), 1, 1 - rows["unearned"])
    lam = np.clip(earned * lr / mean_sev, 0, None)
    claim_alloc = rng.poisson(lam)
    claim_row = np.repeat(np.arange(len(rows)), claim_alloc)
    sigma = 1.2
    sev = rng.lognormal(np.log(mean_sev[claim_row]) - sigma**2 / 2, sigma)
    tail = rng.random(len(sev)) < 0.03
    sev[tail] *= rng.pareto(1.8, tail.sum()) + 1
    rows["claim_alloc"] = claim_alloc
    rows["allocation"] = np.round(np.bincount(claim_row, weights=sev, minlength=len(rows)), 2)
    span = (rows["pol_exp_dt"] - rows["pol_eff_dt"]).dt.days.to_numpy()
    offset = (rng.random(len(rows)) * span).astype(int)
    rows["evt_dt"] = (rows["pol_eff_dt"] + pd.to_timedelta(offset, unit="D")).where(claim_alloc > 0)

    # --- endorsement and cancellation rows (BOP/BMQ) ------------------------
    not_cmq = ~rows["src"].eq("CMQ")
    endo = rows[not_cmq & (rng.random(len(rows)) < 0.10)].copy()
    factor = rng.uniform(-0.15, 0.30, len(endo))
    endo["tx_type_nm"] = "Endorsement"
    endo["tot_wrtn_prm_amt"] = np.round(endo["gross"] * factor, 2)
    endo["expo_amt"] = np.where(factor > 0, np.maximum(np.round(endo["expo_amt"] * factor, 2), 0.01), 0.0)
    canc = rows[not_cmq & rows["pol_stat"].eq("Cancelled")].copy()
    canc["tx_type_nm"] = "Cancellation"
    canc["tot_wrtn_prm_amt"] = np.round(-canc["gross"] * canc["unearned"], 2)
    canc["expo_amt"] = 0.0
    for extra in (endo, canc):
        extra["claim_alloc"] = 0
        extra["allocation"] = 0.0
        extra["evt_dt"] = pd.NaT

    out = pd.concat([rows, endo, canc], ignore_index=True)
    cols = ["src", "pol_num", "pol_eff_dt", "pol_exp_dt", "covg_type_desc", "class1_cd", "expo_amt",
            "expn_bs", "rsk_loc_id", "rsk_itm_id", "loc_st_abbr", "loc_zipcd", "trr_cd", "mm_seg_cd",
            "each_occ_lmt_amt", "genl_ag_lmt_amt", "tot_wrtn_prm_amt",
            "pol_stat", "bi_ded_amt", "pd_ded_amt", "csl_ded_amt", "tx_type_nm", "allocation",
            "claim_alloc", "evt_dt"]
    out = out[cols].sort_values(["src", "pol_num", "rsk_loc_id", "rsk_itm_id", "covg_type_desc"]).reset_index(drop=True)
    out["class1_cd"] = out["class1_cd"].astype("object")
    return out


def build_sot(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prem = (df.assign(pol_yr=df["pol_eff_dt"].dt.year)
              .groupby(["src", "covg_type_desc", "pol_yr"], as_index=False)["tot_wrtn_prm_amt"].sum()
              .rename(columns={"tot_wrtn_prm_amt": "wrtn_prm"}))
    claims = df[df["evt_dt"].notna()]
    loss = (claims.assign(loss_yr=claims["evt_dt"].dt.year)
                  .groupby(["src", "covg_type_desc", "loss_yr"], as_index=False)[["allocation", "claim_alloc"]].sum())
    return prem, loss


def inject_issues(df: pd.DataFrame, sot_prem: pd.DataFrame, rng: np.random.Generator):
    df = df.copy()
    sot_prem = sot_prem.copy()
    pick = lambda idx, k: rng.choice(np.asarray(idx), size=min(k, len(idx)), replace=False)  # noqa: E731
    pol_yr = df["pol_eff_dt"].dt.year

    # recon issues first (they reference clean values)
    liquor = sot_prem["src"].eq("CMQ") & sot_prem["covg_type_desc"].eq("Liquor Liability")
    sot_prem.loc[liquor, "wrtn_prm"] *= 1.03
    df = df[~(df["src"].eq("BOP") & pol_yr.eq(2020) & df["covg_type_desc"].eq("Medical Payments"))].copy()
    loss_yr = df["evt_dt"].dt.year
    df.loc[df["src"].eq("BOP") & loss_yr.eq(2021), "allocation"] *= 0.95
    df.loc[df["src"].eq("BMQ") & loss_yr.eq(2022), "claim_alloc"] *= 2
    pol_yr = df["pol_eff_dt"].dt.year

    # key issues
    bmq_nc = df.index[df["src"].eq("BMQ") & df["claim_alloc"].eq(0)]
    dups = df.loc[pick(bmq_nc, int(0.005 * df["src"].eq("BMQ").sum()))]
    cmq = df[df["src"].eq("CMQ")]
    multi = cmq.groupby("pol_num")["class1_cd"].nunique()
    multi_pols = pick(multi.index[multi > 1], max(1, int(0.01 * len(multi))))
    df.loc[df["pol_num"].isin(multi_pols), "class1_cd"] = ""

    # missing issues
    cmq_idx = df.index[df["src"].eq("CMQ")]
    df.loc[pick(cmq_idx, int(0.12 * len(cmq_idx))), "class1_cd"] = None
    bop_idx = df.index[df["src"].eq("BOP")]
    df.loc[pick(bop_idx, int(0.03 * len(bop_idx))), "loc_zipcd"] = ""
    bmq_idx = df.index[df["src"].eq("BMQ")]
    df.loc[pick(bmq_idx, int(0.02 * len(bmq_idx))), "expn_bs"] = "UNK"
    small_claims = df.index[(df["claim_alloc"] > 0) & (df["allocation"] < df.loc[df["claim_alloc"] > 0, "allocation"].median())]
    df.loc[pick(small_claims, int(0.01 * (df["claim_alloc"] > 0).sum())), "evt_dt"] = pd.NaT

    # distribution issues
    df.loc[df["src"].eq("BMQ") & pol_yr.eq(2019), "tot_wrtn_prm_amt"] *= 100
    base_rows = df.index[df["tx_type_nm"].isin(["New Business", "Renewal"]) & (df["expo_amt"] > 0)]
    neg = pick(base_rows, 20)
    df.loc[neg, "expo_amt"] = -df.loc[neg, "expo_amt"]
    not_bmq19 = df.index[~(df["src"].eq("BMQ") & pol_yr.eq(2019)) & (df["tot_wrtn_prm_amt"] > 0)]
    df.loc[pick(not_bmq19, 10), "tot_wrtn_prm_amt"] *= 1000
    yr23 = df.index[pol_yr.eq(2023) & df["class1_cd"].notna() & df["class1_cd"].ne("")]
    df.loc[pick(yr23, int(0.02 * pol_yr.eq(2023).sum())), "class1_cd"] = "99999"

    # value issues: a limit value only one source uses, and a deductible sentinel
    bmq_idx = df.index[df["src"].eq("BMQ")]
    df.loc[pick(bmq_idx, int(0.01 * len(bmq_idx))), "each_occ_lmt_amt"] = 10_000_000.0
    bop_ded = df.index[df["src"].eq("BOP") & df["bi_ded_amt"].notna()]
    df.loc[pick(bop_ded, int(0.015 * len(bop_ded))), "bi_ded_amt"] = 99_999.0

    # exposure issues
    top_class = df["class1_cd"].value_counts().index[0]
    top_rows = df.index[df["class1_cd"].eq(top_class)]
    current = df.loc[top_rows[0], "expn_bs"]
    other = next(b for b in EXPO_BASES if b != current)
    df.loc[pick(top_rows, int(0.3 * len(top_rows))), "expn_bs"] = other
    pos_prem = df.index[(df["tot_wrtn_prm_amt"] > 0) & (df["expo_amt"] > 0)]
    df.loc[pick(pos_prem, 50), "expo_amt"] = 0.0

    # date issues
    bad_pols = pick(df["pol_num"].unique(), 15)
    m = df["pol_num"].isin(bad_pols)
    df.loc[m, "pol_exp_dt"] = df.loc[m, "pol_eff_dt"] - pd.Timedelta(days=30)
    late_eff = df.index[(df["claim_alloc"] > 0) & df["evt_dt"].notna() & (df["pol_eff_dt"].dt.dayofyear > 40)
                        & (df["allocation"] < df.loc[df["claim_alloc"] > 0, "allocation"].median())
                        & ~m]
    ev = pick(late_eff, 25)
    df.loc[ev, "evt_dt"] = df.loc[ev, "pol_eff_dt"] - pd.Timedelta(days=20)

    df = pd.concat([df, dups], ignore_index=True)
    return df.reset_index(drop=True), sot_prem


def write_outputs(out: Path, tables: dict[str, pd.DataFrame]) -> Path:
    import duckdb

    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "gl_synth.duckdb"
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    for name, frame in tables.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)
        con.register("frame", frame)
        date_cols = [c for c in frame.columns if c.endswith("_dt")]
        select = ", ".join(f'CAST("{c}" AS DATE) AS "{c}"' if c in date_cols else f'"{c}"' for c in frame.columns)
        con.execute(f"CREATE TABLE {name} AS SELECT {select} FROM frame")
        con.unregister("frame")
    con.close()
    return db_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-policies", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--no-inject", action="store_true", help="skip injected issues (clean data)")
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    clean = generate_clean(args.n_policies, rng)
    sot_prem, sot_loss = build_sot(clean)
    df = clean
    if not args.no_inject:
        df, sot_prem = inject_issues(clean, sot_prem, rng)
    path = write_outputs(args.out, {"gl_master_synth": df, "sot_premium_synth": sot_prem, "sot_loss_synth": sot_loss})
    print(f"wrote {len(df):,} rows ({df['src'].value_counts().to_dict()}) -> {path}")


if __name__ == "__main__":
    main()
