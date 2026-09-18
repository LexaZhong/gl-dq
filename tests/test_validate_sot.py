"""The source-of-truth validator, and the paste-able comparison SQL it prints."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "jobs"))

import validate_sot as v  # noqa: E402
from gl_dq.checks._recon import pipeline_agg, reconcile, sot_agg  # noqa: E402


def _compare_sql(ctx, check):
    cfg = ctx.check_config(check)
    dims, measures = v.dims_of(ctx, check, cfg), v.measures_of(ctx, check, cfg)
    sot_sql = ctx.render_user_sql(cfg.sot_query)
    return dims, measures, cfg, sot_sql, ctx.render_sql(
        "recon_compare.sql.j2", dims=dims, measures=list(measures),
        pipeline_sql=ctx.render_sql("agg_by_dims.sql.j2", dims=dims,
                                    measures={a: ctx.schema.ref(p) for a, (p, _) in measures.items()}, where=None),
        sot_sql=ctx.render_sql("agg_sot.sql.j2", sot_sql=sot_sql,
                               dims={d: cfg.dim_map.get(d, d) for d in dims},
                               measures={a: c for a, (_, c) in measures.items()}))


@pytest.mark.parametrize("check", ["premium_recon", "loss_recon"])
def test_validator_accepts_the_configured_queries(ctx_injected, check, capsys):
    assert v.validate(ctx_injected, check) == []
    out = capsys.readouterr().out
    assert "MISS" not in out and "ok    measure" in out


@pytest.mark.parametrize("check", ["premium_recon", "loss_recon"])
def test_printed_sql_reproduces_the_dashboard(ctx_injected, check):
    """The SQL people paste into the SQL editor must give the same answer as the app."""
    dims, measures, cfg, sot_sql, sql = _compare_sql(ctx_injected, check)
    out = ctx_injected.db.query(sql)
    alias = list(measures)[0]
    chk = ctx_injected.make_check(check, cfg)
    tol = cfg.tolerance if check == "premium_recon" else cfg.tolerance_loss
    rec = reconcile(pipeline_agg(chk, dims, {a: ctx_injected.schema.ref(p) for a, (p, _) in measures.items()}),
                    sot_agg(chk, sot_sql, dims, cfg.dim_map, {a: c for a, (_, c) in measures.items()}),
                    dims, alias, tol)
    assert len(out) == len(rec)
    assert out[f"{alias}_diff"].sum() == pytest.approx(rec["diff"].sum())
    assert out[f"{alias}_pipeline"].sum() == pytest.approx(rec["pipeline"].sum())


def test_missing_dimension_is_reported(ctx_injected, tmp_path, capsys):
    store = ctx_injected.config_store
    from gl_dq.core.storage import LocalStorage

    (tmp_path / "sql").mkdir()
    (tmp_path / "sql" / "sot.sql").write_text("SELECT src, wrtn_prm FROM sot_premium_synth", encoding="utf-8")  # no coverage column
    cfg = ctx_injected.check_config("premium_recon").model_copy(update={"sot_query": "sql/sot.sql"})
    ctx_injected.config_store = LocalStorage(tmp_path)
    ctx_injected.config_store.write_text("checks/premium_recon.yaml", cfg.model_dump_json())
    try:
        problems = v.validate(ctx_injected, "premium_recon")
    finally:
        ctx_injected.config_store = store
    assert any("covg_type_desc" in p for p in problems)
    assert "MISS  dimension covg_type_desc" in capsys.readouterr().out
