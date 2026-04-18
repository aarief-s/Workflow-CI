# Workflow-CI — Lufthi Arief Syabana

Repository CI/CD untuk otomatisasi training model Machine Learning menggunakan **MLflow Project** dan **GitHub Actions**.

## 👤 Identitas
- **Nama:** Lufthi Arief Syabana
- **Dataset:** Smartphone Usage and Addiction Analysis
- **Task:** Binary Classification — `addicted_label` (0/1)

## 📁 Struktur Repository

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml                              # GitHub Actions workflow (Advance)
├── MLProject/
│   ├── MLProject                               # File konfigurasi MLflow Project
│   ├── conda.yaml                              # Dependensi environment
│   ├── modelling.py                            # Script training utama
│   ├── DockerHub.txt                           # Link Docker Hub image
│   └── smartphone_usage_preprocessing/
│       ├── train.csv                           # Data training (6.000 rows)
│       ├── test.csv                            # Data testing (1.500 rows)
│       └── smartphone_usage_preprocessing.csv  # Dataset lengkap
└── README.md
```

## ⚙️ GitHub Actions Workflow

Workflow CI terdiri dari **2 jobs** yang berjalan secara berurutan:

### Job 1: `train` — MLflow Project
| Step | Deskripsi |
|---|---|
| Checkout repo | Ambil kode terbaru |
| Setup Python 3.12 | Konfigurasi environment |
| Install dependencies | Install mlflow, sklearn, dll |
| Verify dataset | Cek dataset tersedia |
| Run MLflow Project | `mlflow run .` dengan parameter |
| Verify output | Cek model & artefak dihasilkan |
| Upload artifacts | Upload ke GitHub Artifacts (30 hari) |
| Commit artifacts | Push model ke repo GitHub |

### Job 2: `docker` — Build & Push Docker Hub
| Step | Deskripsi |
|---|---|
| Download artifacts | Ambil hasil dari job train |
| Login Docker Hub | Autentikasi via secrets |
| `mlflow models build-docker` | Build image dari model |
| Push ke Docker Hub | Push `latest` + tag run number |
| Update DockerHub.txt | Update link di repo |

## 🚀 Trigger Workflow

Workflow akan berjalan otomatis ketika:
- **Push** ke `main` yang mengubah file di folder `MLProject/`
- **Manual** via tab Actions → "Run workflow"
- **Terjadwal** setiap Senin pukul 08:00 UTC

## 🔐 Secrets yang Diperlukan

Tambahkan di `Settings → Secrets and variables → Actions`:

| Secret | Keterangan |
|---|---|
| `DOCKERHUB_USERNAME` | Username Docker Hub Anda |
| `DOCKERHUB_TOKEN` | Access Token dari Docker Hub |

Cara membuat Docker Hub Token:
1. Login ke [hub.docker.com](https://hub.docker.com)
2. `Account Settings → Security → New Access Token`
3. Copy token dan simpan sebagai secret di GitHub

## ▶️ Menjalankan MLflow Project Secara Lokal

```bash
cd MLProject

# Dengan parameter default
mlflow run . --env-manager local

# Dengan parameter custom
mlflow run . --env-manager local \
  -P n_estimators=200 \
  -P max_depth=12 \
  -P min_samples_split=5

# Lihat hasil di MLflow UI
mlflow ui
# Buka http://localhost:5000
```

## 🐳 Docker

```bash
# Pull image dari Docker Hub
docker pull <docker_username>/smartphone-addiction-ml:latest

# Jalankan model serving
docker run -p 5001:8080 <docker_username>/smartphone-addiction-ml:latest

# Test inference
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": [...], "data": [[...]]}}'
```

## 📦 Requirements
```
python=3.12.7
mlflow==2.19.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
matplotlib==3.9.2
seaborn==0.13.2
joblib==1.4.2
```
