"""Column transforms: standardize, map values, and the JSON the modelling pipeline reads."""
from dataclasses import replace

import pandas as pd
import pytest

from gl_dq.core.transforms import (
    ColumnTransform,
    Standardize,
    TransformLibrary,
    apply_series,
    load_transforms,
    preprocessing_json,
    save_transforms,
    sql_expr,
)

MESSY = ["  sales ", "SALES", "Sales", "payroll", "UNK"]


def test_standardize_runs_in_one_fixed_order():
    t = ColumnTransform(column="expn_bs_std",
                        standardize=Standardize(trim=True, case="upper", zero_pad=0, cast="none"))
    assert list(apply_series(MESSY, t)) == ["SALES", "SALES", "SALES", "PAYROLL", "UNK"]


def test_zero_pad_and_cast():
    pad = ColumnTransform(column="trr_cd", standardize=Standardize(zero_pad=3))
    assert list(apply_series(["7", "42", "123"], pad)) == ["007", "042", "123"]
    as_int = ColumnTransform(column="x", standardize=Standardize(trim=True, cast="int"))
    cast = apply_series([" 42 ", "7.6", "nope"], as_int)
    assert list(cast[:2]) == [42, 8] and pd.isna(cast[2])


def test_mapping_applies_after_standardizing():
    """One entry has to catch every spelling, which is only true if trim and case run first."""
    t = ColumnTransform(column="expn_bs_std", standardize=Standardize(trim=True, case="upper"),
                        mapping={"SALES": "Sales", "PAYROLL": "Payroll"})
    assert list(apply_series(MESSY, t)) == ["Sales", "Sales", "Sales", "Payroll", "UNK"]


def test_unmapped_values_keep_other_or_null():
    base = dict(column="c", mapping={"A": "Alpha"})
    assert list(apply_series(["A", "Z"], ColumnTransform(**base, unmapped="keep"))) == ["Alpha", "Z"]
    assert list(apply_series(["A", "Z"], ColumnTransform(**base, unmapped="other"))) == ["Alpha", "<other>"]
    dropped = apply_series(["A", "Z"], ColumnTransform(**base, unmapped="null"))
    assert dropped[0] == "Alpha" and pd.isna(dropped[1])


def test_steps_are_ordered_and_self_describing():
    t = ColumnTransform(column="c", standardize=Standardize(trim=True, case="upper", zero_pad=3, cast="string"),
                        mapping={"007": "7"})
    assert [s["op"] for s in t.steps()] == ["trim", "case", "zero_pad", "map_values", "cast"]
    assert t.steps()[3]["mapping"] == {"007": "7"} and t.steps()[3]["unmapped"] == "keep"


def test_inactive_transform_produces_no_steps():
    assert ColumnTransform(column="c").steps() == []
    assert not ColumnTransform(column="c").active


# ---- SQL -------------------------------------------------------------------------------
def test_sql_matches_pandas(ctx_injected):
    """The preview and the query must agree, or the dashboard lies about what it will do."""
    t = ColumnTransform(column="expn_bs_std", standardize=Standardize(trim=True, case="upper"),
                        mapping={"SALES": "Sales"}, unmapped="keep")
    expr = sql_expr(ctx_injected.schema, ctx_injected.dialect, t)
    got = ctx_injected.db.query(
        f"SELECT DISTINCT {expr} AS v FROM gl_master_synth WHERE expn_bs_std IS NOT NULL ORDER BY 1")["v"]
    raw = ctx_injected.db.query(
        "SELECT DISTINCT expn_bs_std AS v FROM gl_master_synth WHERE expn_bs_std IS NOT NULL")["v"]
    assert sorted(got) == sorted(set(apply_series(list(raw), t).dropna()))


def test_sql_escapes_mapped_values(ctx_injected):
    t = ColumnTransform(column="src", mapping={"O'Brien": "OBrien"})
    assert "'O''Brien'" in sql_expr(ctx_injected.schema, ctx_injected.dialect, t)


def test_unknown_column_is_rejected(ctx_injected):
    from gl_dq.core.schema import UnknownColumnError

    with pytest.raises(UnknownColumnError):
        sql_expr(ctx_injected.schema, ctx_injected.dialect, ColumnTransform(column="nope",
                                                                            standardize=Standardize(trim=True)))


# ---- applied to the dashboard ----------------------------------------------------------
def with_transforms(ctx, *transforms, apply=True):
    return replace(ctx, transforms=TransformLibrary(apply_to_dashboard=apply, transforms=list(transforms)))


def test_nothing_applied_is_a_no_op(ctx_injected):
    t = ColumnTransform(column="expn_bs_std", standardize=Standardize(case="upper"))
    off = with_transforms(ctx_injected, t, apply=False)
    assert off.table_expr == off.project.table
    assert off.transforms.fingerprint() == ""


def test_applying_a_transform_changes_what_every_page_sees(ctx_injected):
    from gl_dq.summary import summarize

    t = ColumnTransform(column="expn_bs_std", standardize=Standardize(case="upper"))
    ctx = with_transforms(ctx_injected, t)
    assert "SELECT" in ctx.table_expr and "UPPER" in ctx.table_expr
    bases = set(summarize(ctx, ["expn_bs_std"])["expn_bs_std"].dropna())
    assert bases and all(b == b.upper() for b in bases)
    raw = set(summarize(ctx_injected, ["expn_bs_std"])["expn_bs_std"].dropna())
    assert raw != bases and len(bases) <= len(raw)


def test_a_mapping_collapses_levels_everywhere(ctx_injected):
    from gl_dq.summary import summarize

    t = ColumnTransform(column="expn_bs_std", mapping={"Sales": "Revenue", "Payroll": "Revenue"})
    ctx = with_transforms(ctx_injected, t)
    by_base = summarize(ctx, ["expn_bs_std"])
    assert "Revenue" in set(by_base["expn_bs_std"]) and "Sales" not in set(by_base["expn_bs_std"])
    # the book is unchanged: rows moved between levels, none were lost
    assert by_base["records"].sum() == summarize(ctx_injected, ["expn_bs_std"])["records"].sum()


def test_transforms_sit_inside_the_filters(ctx_injected):
    """A filter must be written against the standardized value, not the source's spelling."""
    from gl_dq.core.filters import Filter, FilterSet

    t = ColumnTransform(column="expn_bs_std", standardize=Standardize(case="upper"))
    ctx = replace(ctx_injected, transforms=TransformLibrary(apply_to_dashboard=True, transforms=[t]),
                  filters=FilterSet(filters=[Filter(key="k", enabled=True, column="expn_bs_std",
                                                    op="in", values=["SALES"])]))
    n = int(ctx.db.query(f"SELECT COUNT(*) n FROM {ctx.table_expr}").iloc[0]["n"])
    raw = int(ctx_injected.db.query(
        "SELECT COUNT(*) n FROM gl_master_synth WHERE UPPER(expn_bs_std) = 'SALES'").iloc[0]["n"])
    assert n == raw > 0


def test_every_check_runs_with_transforms_applied(ctx_injected):
    t = ColumnTransform(column="expn_bs_std", standardize=Standardize(trim=True, case="upper"))
    ctx = with_transforms(ctx_injected, t)
    for name in ctx.enabled_checks():
        ctx.make_check(name).run()


# ---- storage and handover --------------------------------------------------------------
def test_library_round_trips_as_json(tmp_path):
    from gl_dq.core.storage import make_storage

    store = make_storage(str(tmp_path), tmp_path)
    lib = TransformLibrary(apply_to_dashboard=True).put(
        ColumnTransform(column="expn_bs_std", standardize=Standardize(trim=True),
                        mapping={"SALES": "Sales"}, description="agreed vocabulary").stamped("ds@test.com"))
    save_transforms(store, lib)
    assert (tmp_path / "transforms.json").exists()
    back = load_transforms(store)
    assert back == lib and back.get("expn_bs_std").author == "ds@test.com"
    assert back.drop("expn_bs_std").transforms == []


def test_one_transform_per_column():
    with pytest.raises(ValueError):
        TransformLibrary(transforms=[ColumnTransform(column="c"), ColumnTransform(column="c")])
    lib = TransformLibrary().put(ColumnTransform(column="c", mapping={"a": "b"}))
    lib = lib.put(ColumnTransform(column="c", mapping={"a": "z"}))  # replaces
    assert len(lib.transforms) == 1 and lib.get("c").mapping == {"a": "z"}


def test_pipeline_json_merges_transforms_binnings_and_review_steps(ctx_injected, tmp_path):
    """One file for the modelling phase: every column's steps, in the order they must run."""
    from gl_dq.core.binning import BinningLibrary, BinSpec, save_binnings
    from gl_dq.core.knowledge import PreprocessingStep
    from gl_dq.core.storage import make_storage

    ctx = with_transforms(ctx_injected,
                          ColumnTransform(column="expn_bs_std", standardize=Standardize(trim=True, cast="string"),
                                          mapping={"SALES": "Sales"}, description="agreed vocabulary"))
    ctx = replace(ctx, config_store=make_storage(str(tmp_path), tmp_path))  # never write the repo config
    save_binnings(ctx.config_store, BinningLibrary().put(
        BinSpec(variable="each_occ_lmt_amt", name="quartiles", method="custom", cuts=[1e6, 2e6])))
    ctx.knowledge.add_preprocessing_step(
        "expn_bs_std", PreprocessingStep(op="impute", params={"strategy": "constant", "value": "UNK"},
                                         rationale="blank base"), "ds@test.com",
        set_status="preprocess_in_modeling")

    spec = preprocessing_json(ctx)
    assert spec["table"] == ctx.project.table and spec["profile"] == ctx.profile
    ops = [s["op"] for s in spec["columns"]["expn_bs_std"]["steps"]]
    assert ops == ["trim", "map_values", "cast", "impute"], "steps must come out in applicable order"
    assert spec["columns"]["expn_bs_std"]["note"] == "agreed vocabulary"
    binned = spec["columns"]["each_occ_lmt_amt"]["steps"][0]
    assert binned["op"] == "bin" and binned["cuts"] == [1e6, 2e6] and binned["labels"]


def test_shipped_transform_library_is_empty_and_off(ctx_injected):
    lib = load_transforms(ctx_injected.config_store)
    assert lib.transforms == [] and lib.apply_to_dashboard is False
