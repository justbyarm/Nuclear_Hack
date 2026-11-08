# Геометрия (У2): что готово для команды

## У3 — фейки для склейки (можно работать уже сейчас)

```bash
PYTHONPATH=src python -m mct.geometry.fakes --plan calib/plan.json --out outputs/fake --minutes 10 --seed 0
```

В `outputs/fake/`: `tracklets.parquet`, `tracklet_dets.parquet`, `trajectories.parquet`,
`calib/plan.json`, `calib/graph.json` — в формате контрактов. Эталон для проверки:
`gt_tracklets.csv` (колонки `gt_strict` и `gt_transit`) и `gt_summary.json` (сколько треков
должно получиться в каждом режиме). Эталон в склейку не подавать.

Дополнительные колонки `trajectories.parquet` (предложение к контракту):
- `vx, vy` — скорость, м/с (экстраполяция через слепой участок, проверка 2,5 м/с);
- `cams_cover` — JSON-список камер, чья зона обзора накрывает точку. Если трек пропал там, где
  его должна видеть камера, — это пропуск детекции, а не уход;
- `zone` — `exit:<id>` | `portal` | `occluder` | `overlap` | `open` | `outside`.

`graph.json`: `gaps[]` с `a`, `b`, `portal_a/b` (отрезки, метры), `length_m`, `t_min_s`, `t_max_s`
(окно времени прохода) и `doors_inside` (двери внутри слепого участка: переход неоднозначен).
Тупики — `deadend_*`, у них `b = null`.

## У1 — фильтр «на полу» (S1b)

```bash
PYTHONPATH=src python -m mct.geometry.floor_check --detections outputs/run/detections.parquet \
    --calib calib --out outputs/run/floor_check.parquet
```

Требования к `detections.parquet`:
- **все пиксели — в разрешении исходного видео 2592×1944**, даже если YOLO работает на уменьшенном кадре;
- `mask_rle` — COCO RLE JSON-строкой (как pycocotools), `size` = [1944, 2592];
- **время `t` — в секундах.** Камера 4 пишет 20 fps, остальные 25: по номеру кадра камеры не синхронизировать.

Отбрасывать только `on_floor == False`. `in_fov` — для диагностики, по нему не фильтровать.
Камеры без калибровки проходят целиком (`on_floor = True`).

## У3 — траектории S5 (реальные данные)

```bash
PYTHONPATH=src python -m mct.geometry.trajectories --run outputs/run --calib calib
```

Нужно в `outputs/run/`: `detections.parquet` (У1), `tracklet_dets.parquet` (У4), желательно
`floor_check.parquet` (S1b). Пишет `trajectories.parquet` в том же формате, что фейки.

## У1 и UI — карта трека

```python
from mct.geometry.map_view import MapData, track_figure
md = MapData.load("outputs/run", "calib")
fig, path = track_figure(md, global_id)       # st.plotly_chart(fig, use_container_width=True)
```

Страница целиком: `app/pages/3_map.py`. Пока нет `assignments.parquet`, на фейках
берёт эталонные треки — можно проверить уже сейчас:
`streamlit run app/pages/3_map.py` после генерации фейков в `outputs/fake`.

## У2 — когда придут детекции (порядок запуска)

```bash
# 1. фильтр «на полу» (заодно точки ног для всего остального)
PYTHONPATH=src python -m mct.geometry.floor_check --detections outputs/run/detections.parquet --calib calib --out outputs/run/floor_check.parquet
# 2. уточнение калибровки по людям: сначала отчёт, потом --apply
PYTHONPATH=src python -m mct.geometry.refine --run outputs/run --calib calib
PYTHONPATH=src python -m mct.geometry.refine --run outputs/run --calib calib --apply
# 3. D_max, смещения камер, сдвиги времени -> outputs/run/overlap_report.json (D_max -> У3)
PYTHONPATH=src python -m mct.geometry.overlap --run outputs/run --calib calib
# 4. после треклетов У4 — траектории для склейки
PYTHONPATH=src python -m mct.geometry.trajectories --run outputs/run --calib calib
```
