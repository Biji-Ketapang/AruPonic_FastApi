# 🚀 Aruponic Dashboard (Backend)

Repository ini berisi *source code* antarmuka (Backend) untuk proyek **Aruponic**, dibangun menggunakan **React** dan **Vite**. Aplikasi ini berfungsi untuk memvisualisasikan data dan berinteraksi dengan model *Deep Learning* di sisi backend.

## 🛠️ Tech Stack

* **Framework:** React JS
* **Build Tool:** Vite
* **Language:** JavaScript / JSX
* **Backend Integration:** FastAPI (Python)

## 📋 Prasyarat (Prerequisites)

Sebelum memulai, pastikan kamu telah menginstal:
* [Node.js](https://nodejs.org/) (Versi 18 atau lebih baru)
* [Python](https://www.python.org/) (Versi 3.9 atau lebih baru)

---

## 💻 Cara Menjalankan Aplikasi (Backend)

Ikuti langkah-langkah di bawah ini untuk menjalankan aplikasi secara lokal.

### 1. Setup & Jalankan Backend (API)
Frontend ini membutuhkan data dari backend. Jalankan perintah ini di terminal (arahkan ke folder backend/root terlebih dahulu):

```bash
# 1. Buat Virtual Environment (Hanya pertama kali)
python -m venv .venv

# 2. Aktifkan Virtual Environment
# Untuk Windows:
.venv\Scripts\Activate.ps1
# Untuk Mac/Linux:
# source .venv/bin/activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Jalankan Server Backend
uvicorn main:app --reload
# ATAU
fastapi dev main.py
