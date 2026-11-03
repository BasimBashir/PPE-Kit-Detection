# Third-party notices

This project is licensed under AGPL-3.0-or-later (see `LICENSE`). It depends on, but does not redistribute, the
components below. Model weights and datasets are covered in [docs/model-and-data-licenses.md](docs/model-and-data-licenses.md).

## Python (backend/pyproject.toml, versions pinned in uv.lock)
| Package | License |
|---|---|
| ultralytics 8.4.198 (YOLO26 code + weights) | AGPL-3.0 |
| torch / torchvision 2.14.1 (cu130) | BSD-3-Clause |
| fastapi 0.162.0, uvicorn, pydantic, pydantic-settings | MIT / BSD-3-Clause |
| SQLAlchemy 2.1.6, alembic 1.20.1 | MIT |
| psycopg 3.3.6 | LGPL-3.0 |
| av (PyAV) 19.0.2 / FFmpeg libraries | BSD-3-Clause / LGPL-2.1+ (bundled FFmpeg build) |
| numpy, opencv-python (via ultralytics) | BSD-3-Clause / Apache-2.0 |
| prometheus-client 0.26.0 | Apache-2.0 |
| psutil | BSD-3-Clause |
| nvidia-ml-py | BSD-3-Clause |
| argon2-cffi | MIT |
| PyYAML | MIT |

## JavaScript (frontend/package.json, exact versions)
react / react-dom 19.3.1 (MIT), react-router-dom 7.18.5 (MIT), recharts 3.10.1 (MIT); build/test: vite, vitest,
typescript, testing-library, jsdom (MIT / Apache-2.0).

## Services (docker-compose.yml)
PostgreSQL 17 (PostgreSQL License), MediaMTX 1.21.3 (MIT), Toxiproxy 2.13.0 (MIT), Prometheus 3.15.1 (Apache-2.0),
Grafana 13.2.5 (AGPL-3.0).

## Media
Demo clips: Pexels License (downloaded by script, not redistributed). Training data: Hard Hat Workers, CC0-1.0.
