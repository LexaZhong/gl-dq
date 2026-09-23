"""Streamlit session/caching glue."""
from __future__ import annotations

import getpass
import importlib
import os
from dataclasses import replace

import streamlit as st
from pydantic import BaseModel

from gl_dq.core.context import load_context
from gl_dq.core.binning import BinningLibrary, BinningSet, BinSpec, load_binnings
from gl_dq.core.filters import FilterSet
from gl_dq.core.transforms import TransformLibrary, load_transforms


def profile() -> str:
    return os.environ.get("DQ_PROFILE", "synthetic")


@st.cache_resource(show_spinner="Connecting to data…")
def _context(profile_name: str):
    return load_context(profile_name)


def get_context():
    """The context this session works with: one shared connection, this session's global filters."""
    return _with_filters(_context(profile()), session_filters_json())


def _with_filters(ctx, filters_json: str):
    """A view of the context with these filters.

    A copy, never a mutation: `_context` is an `st.cache_resource`, shared across every session, so
    setting `ctx.filters` on it would apply one user's filters to everyone. `replace` keeps the
    same db connection, schema and stores.
    """
    if not filters_json:
        return ctx
    fs = FilterSet.model_validate_json(filters_json)
    return ctx if fs == ctx.filters else replace(ctx, filters=fs)


# ---- per-session global filters ------------------------------------------------------
def saved_filters() -> FilterSet:
    """The filters in config/filters.yaml (what the refresh job uses)."""
    return _context(profile()).filters


def session_filters() -> FilterSet:
    key = "global_filters"
    if key not in st.session_state:
        # a deep copy: `saved_filters()` belongs to the cached Context, which every session shares
        st.session_state[key] = saved_filters().model_copy(deep=True)
    return st.session_state[key]


def set_session_filters(fs: FilterSet) -> None:
    st.session_state["global_filters"] = fs


def reset_session_filters() -> None:
    st.session_state.pop("global_filters", None)


def session_filters_json() -> str:
    """The cache key every filtered computation is keyed on."""
    try:
        return session_filters().model_dump_json()
    except Exception:  # noqa: BLE001  (no session context, e.g. in tests)
        return ""


def current_user() -> str:
    """Databricks Apps forwards the signed-in user's email; locally fall back to $DQ_USER or the OS user."""
    email = None
    try:
        headers = st.context.headers
        email = headers.get("X-Forwarded-Email") or headers.get("X-Forwarded-Preferred-Username")
    except Exception:  # noqa: BLE001  (no request context, e.g. in tests)
        pass
    return email or os.environ.get("DQ_USER") or getpass.getuser()


# ---- per-session check configs (applied but not necessarily saved) ------------------
def session_config(name: str):
    key = f"cfg::{name}"
    if key not in st.session_state:
        st.session_state[key] = get_context().check_config(name)
    return st.session_state[key]


def set_session_config(name: str, cfg) -> None:
    st.session_state[f"cfg::{name}"] = cfg


def reset_session_config(name: str) -> None:
    st.session_state.pop(f"cfg::{name}", None)
    for k in st.session_state.pop(f"widgets::{name}", []):
        st.session_state.pop(k, None)


# ---- column transforms (standardize + value mappings) --------------------------------
def session_transforms() -> TransformLibrary:
    if "transform_library" not in st.session_state:
        st.session_state["transform_library"] = load_transforms(_context(profile()).config_store)
    return st.session_state["transform_library"]


def set_session_transforms(lib: TransformLibrary) -> None:
    st.session_state["transform_library"] = lib


# ---- binning schemes (the library on disk, plus what this session has in play) --------
def session_binnings() -> BinningLibrary:
    """The saved library, copied per session so an unsaved edit is never shared."""
    if "binning_library" not in st.session_state:
        st.session_state["binning_library"] = load_binnings(_context(profile()).config_store)
    return st.session_state["binning_library"]


def set_session_binnings(lib: BinningLibrary) -> None:
    st.session_state["binning_library"] = lib


def current_binning(variable: str) -> BinSpec | None:
    """The scheme this session is looking at for one variable (None = raw levels)."""
    return st.session_state.get("binning_in_play", {}).get(variable)


def set_binning(variable: str, spec: BinSpec | None) -> None:
    in_play = dict(st.session_state.get("binning_in_play", {}))
    if spec is None:
        in_play.pop(variable, None)
    else:
        in_play[variable] = spec
    st.session_state["binning_in_play"] = in_play


def binning_set(variables) -> BinningSet:
    """The schemes in play for these variables, as one cache-keyable model."""
    return BinningSet(specs=[s for s in (current_binning(v) for v in variables) if s is not None])


# ---- cached computations -------------------------------------------------------------
# Every one of these takes `filters_json` as a key argument: the global filters change what the
# query returns, so a result computed under one filter set must not be served under another.
@st.cache_data(ttl=3600, show_spinner=False)
def _run_check(profile_name: str, filters_json: str, name: str, cfg_json: str):
    ctx = _with_filters(_context(profile_name), filters_json)
    cfg = ctx.checks[name].Config.model_validate_json(cfg_json)
    return ctx.make_check(name, cfg).run()


def run_check(name: str, cfg):
    return _run_check(profile(), session_filters_json(), name, cfg.model_dump_json())


def _encode(value):
    if isinstance(value, BaseModel):
        return ("__model__", f"{type(value).__module__}:{type(value).__qualname__}", value.model_dump_json())
    return value


def _decode(value):
    if isinstance(value, tuple) and len(value) == 3 and value[0] == "__model__":
        module, qualname = value[1].split(":")
        cls = importlib.import_module(module)
        for part in qualname.split("."):
            cls = getattr(cls, part)
        return cls.model_validate_json(value[2])
    return value


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_method(profile_name: str, filters_json: str, name: str, cfg_json: str, method: str, kwargs: tuple):
    ctx = _with_filters(_context(profile_name), filters_json)
    cls = ctx.checks[name]
    return getattr(cls(ctx, cls.Config.model_validate_json(cfg_json)), method)(**{k: _decode(v) for k, v in kwargs})


def cached_method(name: str, cfg, method: str, **kwargs):
    """Call a check method with Streamlit caching (kwargs: plain values or pydantic models)."""
    return _cached_method(profile(), session_filters_json(), name, cfg.model_dump_json(), method,
                          tuple(sorted((k, _encode(v)) for k, v in kwargs.items())))


@st.cache_data(ttl=3600, show_spinner=False)
def _summary(profile_name: str, filters_json: str, dims: tuple, where: str | None):
    from gl_dq.summary import summarize

    return summarize(_with_filters(_context(profile_name), filters_json), list(dims), where)


def summary(dims, where: str | None = None, filters_json: str | None = None):
    """Portfolio summary. `filters_json=""` deliberately asks for the unfiltered population."""
    return _summary(profile(), session_filters_json() if filters_json is None else filters_json,
                    tuple(dims), where)


@st.cache_data(ttl=3600, show_spinner=False)
def _distinct_values(profile_name: str, filters_json: str, column: str):
    from gl_dq.summary import distinct_values

    return distinct_values(_with_filters(_context(profile_name), filters_json), column)


def distinct_values(column: str):
    return _distinct_values(profile(), session_filters_json(), column)


@st.cache_data(ttl=3600, show_spinner=False)
def _filter_impact(profile_name: str, filters_json: str):
    from gl_dq.summary import filter_impact

    ctx = _context(profile_name)
    return filter_impact(ctx, FilterSet.model_validate_json(filters_json) if filters_json else None)


def filter_impact(fs: FilterSet | None = None):
    """How much each active filter removes (measured against the unfiltered table)."""
    return _filter_impact(profile(), (fs or session_filters()).model_dump_json())


@st.cache_data(ttl=300, show_spinner=False)
def _latest_findings(profile_name: str, offset: int):
    return _context(profile_name).results.latest(offset)


@st.cache_data(ttl=300, show_spinner=False)
def _runs(profile_name: str):
    return _context(profile_name).results.runs()


def latest_findings(offset: int = 0):
    return _latest_findings(profile(), offset)


def runs():
    return _runs(profile())


def clear_data_caches() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()  # the Context caches config/filters.yaml, so re-read it too
