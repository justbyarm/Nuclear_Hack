"""Карта: путь выбранного трека на плане офиса (доп. пункт ТЗ). Владелец — У2.

Сплошная линия — наблюдаемый участок (цвет = камера), пунктир — предполагаемый путь вне обзора.
Папка прогона и выбранный человек — общие для всего приложения:
  run_dir   = st.session_state["run_dir"]   (выбирается на главной, app.py)
  global_id = st.session_state["global_id"] или ?id=N в адресе (кладёт таблица У4).
Читает из папки прогона: trajectories.parquet, tracklets.parquet, assignments.parquet, tracks.csv.
Калибровка: <run_dir>/calib, если есть, иначе calib/ в корне репозитория.
Отдельно для проверки:  streamlit run app/pages/3_map.py
"""
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Карта трека", layout="wide")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mct.geometry.map_view import MapData, track_figure  # noqa: E402

st.title("Путь трека на плане офиса")

run = Path(st.session_state.get("run_dir", "outputs/u1_v3"))
if not run.is_absolute() and not run.exists():
    run = ROOT / run
if "run_dir" not in st.session_state and not run.exists():
    # Страница открыта сама по себе (без главной): берём самый свежий прогон с траекториями.
    cand = sorted((p for p in (ROOT / "outputs").glob("*") if (p / "assignments.parquet").exists()),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if cand:
        run = cand[0]
need = ["trajectories.parquet", "tracklets.parquet", "assignments.parquet"]
missing = [f for f in need if not (run / f).exists()]
if missing:
    st.warning(f"В папке прогона {run} нет: {', '.join(missing)}. Выберите прогон на главной странице.")
    st.stop()
calib = run / "calib" if (run / "calib" / "plan.json").exists() else ROOT / "calib"


@st.cache_resource(show_spinner="Загружаю прогон…")
def load(run_dir: str, calib_dir: str):
    return MapData.load(run_dir, calib_dir, repo_root=ROOT)


md = load(str(run), str(calib))
ids = md.global_ids()
if not ids:
    st.warning("В assignments.parquet нет ни одного трека.")
    st.stop()


def _as_id(v):
    """global_id из session_state / адреса (строка) -> тот же тип, что в assignments."""
    if v is None:
        return None
    for g in ids:
        if str(g) == str(v):
            return g
    return None


gid = _as_id(st.session_state.get("global_id"))
if gid is None:
    gid = _as_id(st.query_params.get("id"))
with st.sidebar:
    gid = st.selectbox("Трек (global_id)", ids, index=ids.index(gid) if gid in ids else 0)
    show_fov = st.checkbox("Зоны обзора камер", value=True)
st.session_state["global_id"] = gid

tp = md.path(gid)
t = None
if tp.observed and tp.t_end > tp.t_start:
    t = st.slider("Время, с", float(tp.t_start), float(tp.t_end), float(tp.t_start), step=0.1)
fig, tp = track_figure(md, gid, t_now=t, show_fov=show_fov)
st.plotly_chart(fig, use_container_width=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Начало, с", f"{tp.t_start:.1f}")
c2.metric("Конец, с", f"{tp.t_end:.1f}")
c3.metric("Камеры", ", ".join(tp.cameras) or "—")
c4.metric("Вне обзора, с", f"{sum(g['t1'] - g['t0'] for g in tp.gaps):.1f}")
