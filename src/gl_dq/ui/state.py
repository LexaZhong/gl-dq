"""Streamlit session/caching glue."""
from __future__ import annotations

import getpass
import importlib
import os

import streamlit as st
from pydantic import BaseModel

from gl_dq.core.context import load_context


def profile() -> str:
    return os.environ.get("DQ_PROFILE", "synthetic")


@st.cache_resource(show_spinner="Connecting to data…")
def _context(profile_name: str):
    return load_context(profile_name)


def get_context():
    return _context(profile())


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


# ---- cached computations -------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _run_check(profile_name: str, name: str, cfg_json: str):
    ctx = _context(profile_name)
    cfg = ctx.checks[name].Config.model_validate_json(cfg_json)
    return ctx.make_check(name, cfg).run()


def run_check(name: str, cfg):
    return _run_check(profile(), name, cfg.model_dump_json())


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
def _cached_method(profile_name: str, name: str, cfg_json: str, method: str, kwargs: tuple):
    ctx = _context(profile_name)
    cls = ctx.checks[name]
    return getattr(cls(ctx, cls.Config.model_validate_json(cfg_json)), method)(**{k: _decode(v) for k, v in kwargs})


def cached_method(name: str, cfg, method: str, **kwargs):
    """Call a check method with Streamlit caching (kwargs: plain values or pydantic models)."""
    return _cached_method(profile(), name, cfg.model_dump_json(), method,
                          tuple(sorted((k, _encode(v)) for k, v in kwargs.items())))


@st.cache_data(ttl=3600, show_spinner=False)
def _summary(profile_name: str, dims: tuple, where: str | None):
    from gl_dq.summary import summarize

    return summarize(_context(profile_name), list(dims), where)


def summary(dims, where: str | None = None):
    return _summary(profile(), tuple(dims), where)


@st.cache_data(ttl=3600, show_spinner=False)
def _distinct_values(profile_name: str, column: str):
    from gl_dq.summary import distinct_values

    return distinct_values(_context(profile_name), column)


def distinct_values(column: str):
    return _distinct_values(profile(), column)


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
