# 🎥 Real-Time Multi-Camera Crowd Behavior Analysis using Deep Learning

This project analyzes **crowd behavior in real time** using multiple surveillance camera feeds.  
Instead of tracking individuals, it focuses on **group-level spatial and temporal dynamics** to detect abnormal crowd behavior.

---

## 🔍 Key Features
- Multi-camera crowd behavior analysis
- Spatio-temporal modeling using deep learning
- Anomaly detection based on reconstruction error
- Interactive Streamlit dashboard
- FastAPI backend deployed on cloud (Render)
- MJPEG-based video streaming for smooth playback

---

## 🏗️ High-Level Architecture

Local Camera Videos (AVI)
↓
FastAPI Backend (OpenCV + MJPEG)
↓
Anomaly Detection (ConvLSTM Autoencoder)
↓
Results Aggregation (JSON)
↓
Streamlit Dashboard (Local)

---

## 📁 Repository Structure
Crowd Behaviour Analysis/
├── backend/
├── frontend/
├── inference/
├── models/
├── results/
├── data/ (not tracked)
├── requirements.txt
└── README.md

---

## 🚀 Quick Start

### Start Backend
```bash
uvicorn backend.app:app --reload
```
### Start Frontend
```bash
streamlit run frontend/streamlit_app.py
```
