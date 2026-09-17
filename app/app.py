"""GL master data cleaning tracker: Streamlit entry point.

Local:      DQ_PROFILE=synthetic streamlit run app/app.py
Databricks: deployed as a Databricks App (see app.yaml / databricks.yml), DQ_PROFILE=prod
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from gl_dq.ui import state, theme  # noqa: E402,F401
from gl_dq.ui.components import check_page  # noqa: E402
from gl_dq.ui.pages import knowledge_page, preprocessing_page, summary_page, tracker_page  # noqa: E402

st.set_page_config(page_title="GL Data Cleaning Tracker", page_icon="🧹", layout="wide")

try:
    ctx = state.get_context()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load profile '{state.profile()}': {e}")
    st.markdown("For local synthetic data run `python synthetic/generate.py --out data/` first.")
    st.stop()

with st.sidebar:
    st.markdown(f"### 🧹 {ctx.project.name}")
    st.caption(f"Signed in as {state.current_user()}")


def _page_fn(name):
    def page():
        check_page(name)

    page.__name__ = f"check_{name}"
    return page


start = os.environ.get("DQ_START_PAGE", "summary")  # default landing page (also used by tests)
checks = ctx.enabled_checks()
if start not in ["summary", "tracker", "preprocessing", "knowledge"] + checks:
    start = "summary"
pages = {
    "Overview": [
        st.Page(summary_page, title="Portfolio summary", icon="📋", url_path="summary", default=start == "summary"),
        st.Page(tracker_page, title="Cleaning tracker", icon="🧭", url_path="tracker", default=start == "tracker"),
        st.Page(preprocessing_page, title="Preprocessing", icon="🧰", url_path="preprocessing",
                default=start == "preprocessing"),
        st.Page(knowledge_page, title="Knowledge base", icon="📚", url_path="knowledge", default=start == "knowledge"),
    ],
    "Checks": [
        st.Page(_page_fn(n), title=ctx.checks[n].title, icon=ctx.checks[n].icon, url_path=n, default=start == n)
        for n in checks
    ],
}
st.navigation(pages).run()
