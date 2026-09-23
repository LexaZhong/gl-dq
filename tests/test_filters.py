"""Global filters: safe predicates, the filtered table expression, impact, and run provenance."""
from dataclasses import replace

import pytest

from gl_dq.core.filters import Filter, FilterSet, load_filters, predicate, save_filters
from gl_dq.core.filters import validate as validate_filter
from gl_dq.core.schema import UnknownColumnError
from gl_dq.summary import filter_impact, summarize

ZERO_EXPO = Filter(key="no_zero_expo", label="Exclude zero exposure", enabled=True,
                   column="expo_amt", op="gt", values=["0"])
BOP_ONLY = Filter(key="bop_only", label="BOP only", enabled=True, column="src", op="in", values=["BOP"])


def with_filters(ctx, *filters):
    return replace(ctx, filters=FilterSet(filters=list(filters)))


# ---- predicates ----------------------------------------------------------------------
def test_structured_predicates(ctx_injected):
    p = predicate(ctx_injected, BOP_ONLY)
    assert p == """CAST("src" AS STRING) IN ('BOP')"""
    assert predicate(ctx_injected, ZERO_EXPO) == '"expo_amt" > 0'  # compared as a number, not text
    assert predicate(ctx_injected, Filter(key="k", column="loc_st_abbr", op="is_null")) == '"loc_st_abbr" IS NULL'


def test_values_are_escaped(ctx_injected):
    f = Filter(key="k", column="src", op="in", values=["O'Brien"])
    assert predicate(ctx_injected, f) == """CAST("src" AS STRING) IN ('O''Brien')"""


def test_null_handling_in_and_not_in(ctx_injected):
    keep_null = Filter(key="k", column="loc_st_abbr", op="in", values=["NY", "<null>"])
    assert "IS NULL" in predicate(ctx_injected, keep_null)
    drop_null = Filter(key="k", column="loc_st_abbr", op="not_in", values=["PR", "<null>"])
    p = predicate(ctx_injected, drop_null)
    assert "IS NOT NULL" in p and "NOT (" in p


def test_unknown_column_is_rejected(ctx_injected):
    with pytest.raises(UnknownColumnError):
        predicate(ctx_injected, Filter(key="k", column="not_a_column", op="is_null"))


def test_a_filter_needs_exactly_one_of_column_or_expr():
    with pytest.raises(ValueError):
        Filter(key="k")
    with pytest.raises(ValueError):
        Filter(key="k", column="src", op="in", values=["BOP"], expr="1=1")
    with pytest.raises(ValueError):
        Filter(key="Bad Key", column="src", op="in", values=["BOP"])
    with pytest.raises(ValueError):  # a comparison takes one value
        Filter(key="k", column="expo_amt", op="gt", values=["0", "1"])


def test_raw_expr_sees_the_unfiltered_table(ctx_injected):
    """A rule that queries the table itself must not be defined in terms of its own result."""
    f = Filter(key="k", enabled=True, expr="src IN (SELECT DISTINCT src FROM {{ raw_table }})")
    p = predicate(ctx_injected, f)
    assert ctx_injected.project.table in p and "SELECT * FROM" not in p


def test_validate_rejects_broken_sql(ctx_injected):
    validate_filter(ctx_injected, ZERO_EXPO)
    with pytest.raises(Exception):
        validate_filter(ctx_injected, Filter(key="k", expr="this is not sql"))


# ---- where / table_expr --------------------------------------------------------------
def test_nothing_active_is_a_no_op(ctx_injected):
    off = with_filters(ctx_injected, ZERO_EXPO.model_copy(update={"enabled": False}))
    assert off.filters.where(off) is None
    assert off.table_expr == off.project.table  # the bare table name: no subquery, no cost
    assert off.filters.fingerprint(off) == ""


def test_table_expr_wraps_when_active(ctx_injected):
    on = with_filters(ctx_injected, ZERO_EXPO)
    assert on.table_expr.startswith(f"(SELECT * FROM {on.project.table} WHERE ")
    assert on.table_expr.endswith(") AS gl")


def test_null_predicate_excludes_the_row(ctx_injected):
    """A rule keeps rows it is TRUE for: a NULL state must not survive 'US states only'."""
    f = Filter(key="us", enabled=True, column="loc_st_abbr", op="not_in", values=["PR"])
    on = with_filters(ctx_injected, f)
    left = ctx_injected.db.query(f"SELECT COUNT(*) n FROM {on.table_expr}").iloc[0]["n"]
    nulls = ctx_injected.db.query(
        f"SELECT COUNT(*) n FROM {ctx_injected.project.table} WHERE loc_st_abbr IS NULL").iloc[0]["n"]
    total = ctx_injected.db.query(f"SELECT COUNT(*) n FROM {ctx_injected.project.table}").iloc[0]["n"]
    assert int(left) <= int(total) - int(nulls)


def test_profiles_scope_a_filter(ctx_injected):
    elsewhere = ZERO_EXPO.model_copy(update={"profiles": ["some_other_profile"]})
    ctx = with_filters(ctx_injected, elsewhere)
    assert ctx.filters.where(ctx) is None and ctx.table_expr == ctx.project.table


def test_fingerprint_is_stable_and_sensitive(ctx_injected):
    a = with_filters(ctx_injected, ZERO_EXPO, BOP_ONLY)
    b = with_filters(ctx_injected, BOP_ONLY, ZERO_EXPO)  # order must not matter
    assert a.filters.fingerprint(a) == b.filters.fingerprint(b)
    changed = with_filters(ctx_injected, ZERO_EXPO.model_copy(update={"values": ["1"]}), BOP_ONLY)
    assert changed.filters.fingerprint(changed) != a.filters.fingerprint(a)
    assert a.filters.fingerprint(a).startswith("bop_only+no_zero_expo#")


# ---- the filter actually filters ------------------------------------------------------
def test_every_query_sees_the_filtered_population(ctx_injected):
    ctx = with_filters(ctx_injected, BOP_ONLY)
    total = summarize(ctx_injected, []).iloc[0]
    kept = summarize(ctx, []).iloc[0]
    bop = ctx_injected.db.query(
        f"SELECT COUNT(*) n, SUM(tot_wrtn_prm_amt) p FROM {ctx.project.table} WHERE src = 'BOP'").iloc[0]
    assert int(kept.records) == int(bop["n"]) < int(total.records)
    assert float(kept.premium) == pytest.approx(float(bop["p"]))
    assert summarize(ctx, ["src"])["src"].tolist() == ["BOP"]


def test_impact_matches_what_the_filter_removes(ctx_injected):
    ctx = with_filters(ctx_injected, ZERO_EXPO, BOP_ONLY)
    imp = filter_impact(ctx)
    total_rows = imp.attrs["rows_total"]
    assert total_rows == int(summarize(ctx_injected, []).iloc[0].records)
    kept = int(summarize(ctx, []).iloc[0].records)
    combined = imp[imp["key"] == "TOTAL"].iloc[0]
    assert total_rows - int(combined["rows_removed"]) == kept       # the total is the one that counts
    per_rule = imp[imp["key"] != "TOTAL"]["rows_removed"].sum()
    assert per_rule >= int(combined["rows_removed"])                # each rule measured on its own
    assert (imp["pct_rows"] <= 1).all()


def test_disabled_rules_are_costed_too(ctx_injected):
    """You have to see what a rule would remove before deciding to turn it on."""
    ctx = with_filters(ctx_injected, BOP_ONLY.model_copy(update={"enabled": False}))
    imp = filter_impact(ctx)
    assert set(imp["key"]) == {"bop_only", "TOTAL"}
    assert int(imp[imp["key"] == "bop_only"].iloc[0]["rows_removed"]) > 0
    assert int(imp[imp["key"] == "TOTAL"].iloc[0]["rows_removed"]) == 0  # nothing active removes nothing


def test_every_check_runs_under_a_filter(ctx_injected):
    """The filtered table is a subquery: every packaged template has to survive it."""
    ctx = with_filters(ctx_injected, ZERO_EXPO)
    for name in ctx.enabled_checks():
        res = ctx.make_check(name).run()
        assert not res.findings.empty or name in ("business_rules",), name
        assert all(ctx.project.table in sql for sql in res.sql.values() if "FROM" in sql)


def test_distribution_excluded_count_follows_the_filter(ctx_injected):
    ctx = with_filters(ctx_injected, BOP_ONLY)
    chk = ctx.make_check("distribution")
    spec = next(v for v in chk.cfg.variables if chk.kind(v) == "numeric")
    from gl_dq.checks.distribution import LogScale

    # model_copy(update=) skips validation, so build the nested model itself
    chk.histogram(spec.model_copy(update={"log_scale": LogScale(method="log10")}))
    sql = chk._sql[f"{spec.name} excluded"]
    assert "WHERE" in sql and "AS gl" in sql  # counted against the filtered population


# ---- storage and provenance -----------------------------------------------------------
def test_filters_round_trip_through_the_config_store(ctx_injected, tmp_path):
    from gl_dq.core.storage import make_storage

    store = make_storage(str(tmp_path), tmp_path)
    fs = FilterSet(filters=[ZERO_EXPO, Filter(key="raw", expr="1 = 1", description="x")])
    save_filters(store, fs)
    assert load_filters(store) == fs
    assert load_filters(make_storage(str(tmp_path / "empty"), tmp_path)) == FilterSet()


def test_the_run_records_its_filters(ctx_injected):
    from gl_dq.runner import refresh

    ctx = with_filters(ctx_injected, BOP_ONLY)
    run_id, findings, errors = refresh(ctx, ["key_uniqueness"], log=lambda *_: None)
    runs = ctx.results.runs()
    assert runs.iloc[-1]["filters"] == ctx.filters.fingerprint(ctx)
    stored = ctx.results.latest()
    assert set(stored["filters"]) == {ctx.filters.fingerprint(ctx)}


def test_shipped_filters_are_all_off(ctx_injected):
    """Turning one on changes every number on every page, so it is never the default."""
    fs = load_filters(ctx_injected.config_store)
    assert fs.filters, "config/filters.yaml should ship the worked examples"
    assert not fs.active(ctx_injected.profile)
    for f in fs.filters:
        assert f.description, f"{f.key} needs to say why it exists"


# ---- reference lists (a CSV of ids) ---------------------------------------------------
def test_expr_can_read_a_reference_csv(ctx_injected, tmp_path):
    """A rule can match against a list kept outside the pipeline: `{{ var }}` comes from sql_vars,
    so the same rule reads a CSV locally and a table (or read_files) on Databricks."""
    csv = tmp_path / "keep_classes.csv"
    # non-blank ids only: a blank line in a CSV reads back as NULL, never as an empty string
    classes = [str(c) for c in ctx_injected.db.query(
        "SELECT DISTINCT class_cd_std FROM gl_master_synth WHERE class_cd_std <> '' ORDER BY 1 LIMIT 3")["class_cd_std"]]
    csv.write_text("class_cd_std\n" + "\n".join(classes) + "\n", encoding="utf-8")

    project = ctx_injected.project.model_copy(
        update={"sql_vars": {**ctx_injected.project.sql_vars,
                             "class_list": f"read_csv_auto('{csv.as_posix()}')"}})
    ctx = replace(ctx_injected, project=project, filters=FilterSet(filters=[Filter(
        key="listed_classes", enabled=True,
        expr="CAST(class_cd_std AS STRING) IN (SELECT CAST(class_cd_std AS STRING) FROM {{ class_list }})")]))

    assert str(csv) in predicate(ctx, ctx.filters.filters[0])
    got = set(summarize(ctx, ["class_cd_std"])["class_cd_std"].astype(str))
    assert got == set(classes)                      # exactly the listed ids survive
    assert summarize(ctx, []).iloc[0].records < summarize(ctx_injected, []).iloc[0].records


def test_a_missing_sql_var_is_named_in_the_error(ctx_injected):
    f = Filter(key="k", enabled=True, expr="id IN (SELECT id FROM {{ nope }})")
    with pytest.raises(Exception, match="nope"):
        predicate(ctx_injected, f)


# ---- exclude_when: an exclusion must not drop what it cannot evaluate ------------------
@pytest.fixture
def three_valued(ctx_injected, tmp_path):
    """A table with the two unknowns that matter: a PCO row with no id, and an unknown coverage."""
    from gl_dq.core.db import ParquetDatabase
    from gl_dq.core.schema import TableSchema

    (tmp_path / "pco_demo.csv").write_text(
        "covg_type_desc,gl_bop_id,premium\n"
        "ProductsCompletedOps,A,10\nProductsCompletedOps,B,20\nProductsCompletedOps,,500\n"
        "Premises/Operations,A,400\n,A,300\n,B,60\n", encoding="utf-8")
    (tmp_path / "pco_list.csv").write_text("gl_bop_id\nA\n", encoding="utf-8")
    db = ParquetDatabase({"pco_demo": str(tmp_path / "pco_demo.csv"), "pco_list": str(tmp_path / "pco_list.csv")})
    project = ctx_injected.project.model_copy(update={"table": "pco_demo"})
    return replace(ctx_injected, project=project, db=db,
                   schema=TableSchema("pco_demo", db.describe("pco_demo"), {}, db.dialect))


IN_LIST = """covg_type_desc = 'ProductsCompletedOps'
             AND CAST(gl_bop_id AS STRING) IN (SELECT CAST(gl_bop_id AS STRING) FROM pco_list)"""


def _kept(ctx):
    return float(ctx.db.query(f"SELECT COALESCE(SUM(premium), 0) AS p FROM {ctx.table_expr}").iloc[0]["p"])


def test_exclude_when_drops_only_what_is_definitely_true(three_valued):
    """Regression: written as a keep-predicate this removed 810 of 1290 instead of 10, because
    `NULL IN (...)` is NULL and an unknown predicate excludes."""
    ctx = replace(three_valued, filters=FilterSet(filters=[
        Filter(key="pco", enabled=True, exclude_when=IN_LIST)]))
    assert _kept(ctx) == 1280.0                      # 1290 total, only the 10 row goes

    as_keep = replace(three_valued, filters=FilterSet(filters=[
        Filter(key="pco", enabled=True, expr=f"NOT ({IN_LIST})")]))
    assert _kept(as_keep) == 480.0                   # the trap `exclude_when` exists to avoid


def test_exclude_when_renders_null_safe(ctx_injected):
    f = Filter(key="k", enabled=True, exclude_when="src = 'BOP'")
    assert predicate(ctx_injected, f) == "NOT COALESCE((src = 'BOP'), FALSE)"
    assert f.summary() == "SQL (drop when true)"


def test_a_filter_sets_exactly_one_of_the_three_forms():
    with pytest.raises(ValueError):
        Filter(key="k", expr="1=1", exclude_when="1=1")
    with pytest.raises(ValueError):
        Filter(key="k", column="src", op="in", values=["BOP"], exclude_when="1=1")
    assert Filter(key="k", exclude_when="1=1").exclude_when


def test_fingerprint_tracks_exclude_when(ctx_injected):
    a = replace(ctx_injected, filters=FilterSet(filters=[Filter(key="k", enabled=True, exclude_when="src = 'BOP'")]))
    b = replace(ctx_injected, filters=FilterSet(filters=[Filter(key="k", enabled=True, exclude_when="src = 'BMQ'")]))
    assert a.filters.fingerprint(a) != b.filters.fingerprint(b)
