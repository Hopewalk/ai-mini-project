# Thailand Land Appraisal Analytics

ทำนายและอธิบายราคาประเมินที่ดิน (กรมธนารักษ์) ด้วย Regression · Classification · Clustering · Bayesian Network
รายละเอียดการออกแบบและผลลัพธ์: [spec.md](spec.md)

## Tooling

| เครื่องมือ | ใช้ทำอะไร | ไฟล์ |
|---|---|---|
| [Poetry](https://python-poetry.org) 2.x | Python dependencies + virtualenv (`.venv` ในโปรเจกต์) | `pyproject.toml`, `poetry.lock`, `poetry.toml` |
| [pnpm](https://pnpm.io) | Frontend assets (Bootstrap, Leaflet, ฟอนต์ IBM Plex Sans Thai) → `app/static/vendor` + task shortcuts | `package.json`, `pnpm-lock.yaml` |

ติดตั้ง (ครั้งแรกครั้งเดียว, ไม่ต้องใช้สิทธิ์ admin):

```powershell
# Windows PowerShell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
Invoke-WebRequest https://get.pnpm.io/install.ps1 -UseBasicParsing | Invoke-Expression
pnpm env use --global lts      # Node.js ผ่าน pnpm (เปิด terminal ใหม่ก่อน)
```
```bash
# macOS / Linux
curl -sSL https://install.python-poetry.org | python3 -
curl -fsSL https://get.pnpm.io/install.sh | sh -
pnpm env use --global lts
```

## Quick start

```powershell
# 1) dependencies
poetry install          # Python (+ dev group: pytest, ipykernel)
pnpm install            # frontend -> app/static/vendor (postinstall)

# 2) ดาวน์โหลดข้อมูล + เตรียม cells + เทรนทั้ง 4 โมเดล  (~1 นาทีหลังดาวน์โหลดเสร็จ)
pnpm train:prepare      # = poetry run python -m ml.train_all --download --prepare

# 3) โหลดเข้า MongoDB (ต้องมี Mongo รันอยู่ เช่น `docker compose up -d mongo`)
pnpm load:mongo         # = poetry run python -m scripts.load_mongo --parcels

# 4) เว็บ (http://127.0.0.1:5000)
pnpm dev                # = poetry run flask --app run run --debug

# 5) tests (test_views ใช้ MongoDB จริง db `landdb_test` — ถ้าไม่มี Mongo จะ skip)
pnpm test               # = poetry run pytest
```

| pnpm script | คำสั่งจริง |
|---|---|
| `pnpm build` | สร้าง `app/static/vendor` ใหม่ (`scripts/build-vendor.mjs`) |
| `pnpm dev` / `pnpm test` | `poetry run flask ... --debug` / `poetry run pytest` |
| `pnpm train` / `pnpm train:prepare` | `poetry run python -m ml.train_all [--download --prepare]` |
| `pnpm load:mongo` | `poetry run python -m scripts.load_mongo --parcels` |
| `pnpm docker:up` / `pnpm docker:down` | `docker compose up -d --build web` / `docker compose stop` |

คำสั่ง Python อื่น ๆ ด้านล่างรันผ่าน `poetry run ...` หรือเข้า shell ด้วย `poetry env activate` ก่อน
เพิ่ม dependency: `poetry add <pkg>` (Python) · `pnpm add <pkg>` (frontend — แล้วเพิ่ม path ใน `scripts/build-vendor.mjs`)

### เทรนโมเดล (ดูความคืบหน้าแบบ live)

```powershell
.\scripts\train.ps1                     # Windows — เทรนทั้ง 4 โมเดล
.\scripts\train.ps1 --prepare           # สร้าง cells + OSM features ใหม่ก่อนเทรน
.\scripts\train.ps1 --only reg,bn       # เทรนบางโมเดล (reg, clf, cluster, bn) — metrics.json ถูก merge
.\scripts\train.ps1 -Load               # เทรนเสร็จแล้วโหลดเข้า Mongo + restart web container
./scripts/train.sh                      # macOS / Linux (LOAD=1 ./scripts/train.sh = แบบ -Load)
pnpm train                              # เหมือน train.ps1 ไม่มี option
```

ระหว่างเทรนจะเห็น: หัวข้อแต่ละขั้น [1/4]…[4/4] + เวลา · progress bar ของทุก CV fit (GridSearchCV, calibration, permutation importance) · ผล CV/test ของทุกโมเดลทันทีที่เทรนเสร็จ · ตารางสรุปตอนจบ
Log ทั้งหมด (ไม่มี progress bar) บันทึกที่ `ml/artifacts/<version>/train.log` · ปิด progress bar: `--no-progress`

Option อื่นของ `python -m ml.train_all`: `--provinces 10,90` · `--version v3` · `--download`
(`--prepare` สร้างข้อมูล cells ใหม่ซึ่งใช้ร่วมกันทุก version — ถ้าเปลี่ยนจังหวัดแล้วต้องเทรนทุก version ที่ใช้ใหม่)

### เลือกจังหวัด

| ค่า `LAND_PROVINCES` | ผล |
|---|---|
| (ว่าง) | 5 จังหวัดนำร่อง: 10 กรุงเทพฯ, 14 อยุธยา, 50 เชียงใหม่, 83 ภูเก็ต, 90 สงขลา |
| `all` | ทั้ง 77 จังหวัด (~1.44 GB) |
| `10,90` | เฉพาะจังหวัดที่ระบุ (รหัสจังหวัด) |

```powershell
$env:LAND_PROVINCES = "all"; pnpm train:prepare
```

### รันทีละขั้น (`poetry run python -m ...`)

| คำสั่ง | Output |
|---|---|
| `python -m ml.download [--skip-osm]` | `ml/data/raw/land/*.csv`, `ml/data/raw/osm/thailand-latest-free.gpkg` |
| `python -m ml.build_cells` | `ml/data/interim/parcels.parquet`, `cells.parquet` + รายงานการถอดรหัสระวาง |
| `python -m ml.geo_features` | `ml/data/processed/cells_features.parquet` |
| `python -m ml.train_regression` | `regressor.joblib`, `regression_residuals.png` |
| `python -m ml.train_classifier` | `classifier.joblib`, `classifier_confusion.png` |
| `python -m ml.train_cluster` | `cluster.joblib`, `cluster_dendrogram.png`, `cluster_map.png` |
| `python -m ml.train_bn` | `bn.joblib`, `bn_dag_expert.png`, `bn_dag_hillclimb.png` |
| `python -m ml.train_all` | ทั้ง 4 โมเดล + `metrics.json` |

Artifacts อยู่ที่ `ml/artifacts/<MODEL_VERSION>/` (default `v1`)

## Docker

ต้องเปิด Docker Desktop ก่อน · image เป็น multi-stage: `node:24-slim` + pnpm สร้าง vendor assets → `python:3.13-slim` + Poetry ติดตั้งตาม `poetry.lock`

```powershell
# ครั้งแรก: เตรียมข้อมูล + เทรน + โหลดเข้า Mongo
docker compose up -d mongo                                              # MongoDB :27017
docker compose run --rm ml python -m ml.train_all --download --prepare  # pipeline ใน container
docker compose run --rm ml python -m scripts.load_mongo --parcels       # cells + model_runs + รายแปลง

# เว็บ (gunicorn) -> http://localhost:8000
docker compose up -d --build web

docker compose logs -f web    # ดู log
docker compose stop           # ปิด (ข้อมูล Mongo ยังอยู่ใน volume)
docker compose down -v        # ลบทั้งหมดรวมข้อมูล
```

หลังเทรนโมเดลใหม่ ให้รัน `load_mongo` แล้ว `docker compose restart web` เพื่อโหลดโมเดลใหม่

- `ml/data` และ `ml/artifacts` mount เป็น volume → ใช้ข้อมูล/โมเดลร่วมกับ local ได้
- ถ้าเทรนบน local แล้ว ข้ามขั้น train ได้ รันแค่ `load_mongo`
- โหลดเข้า Mongo จาก local โดยตรง: `pnpm load:mongo` (ใช้ `MONGO_URI` default `mongodb://localhost:27017/landdb`)

## หน้าเว็บ

| URL | หน้าที่ |
|---|---|
| `/` | ปักหมุดบนแผนที่ (+ เนื้อที่ / ราคาเสนอขาย) → วิเคราะห์ทั้ง 4 โมเดล → `/result/<id>` |
| `/lookup` | ค้นหาราคาประเมินรายแปลงด้วยระวาง + เลขที่ดิน |
| `/map` | แผนที่ช่อง 2×2 กม. ระบายสีตามเกรด / กลุ่มทำเล / ราคา |
| `/bayes` | ถาม Bayesian Network แบบ what-if (predictive / diagnostic / explaining away) |
| `/dashboard` | metrics + กราฟของทุกโมเดล |
| `/history` | ประวัติการวิเคราะห์ (collection `predictions`) |
| `/api/cells?bbox=w,s,e,n` | JSON ของช่องในกรอบแผนที่ |

โครงสร้าง: `app/models` (Mongo data access) · `app/forms` (Flask-WTF) · `app/views` (Blueprints) · `app/services` (ML + geo)

## Data sources

- ราคาประเมินที่ดิน — กรมธนารักษ์ ([catalog.treasury.go.th](https://catalog.treasury.go.th/dataset/land-valuation)), Open Data Common
- OpenStreetMap Thailand — [Geofabrik](https://download.geofabrik.de/asia/thailand.html), © OpenStreetMap contributors (ODbL)
