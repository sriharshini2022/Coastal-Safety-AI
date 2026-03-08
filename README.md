# 🌊 Coastal Safety AI -- Rip Current & Human Risk Detection

![Python](https://img.shields.io/badge/Python-3.x-blue) ![Computer
Vision](https://img.shields.io/badge/Computer%20Vision-YOLOv8-orange)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)
![Status](https://img.shields.io/badge/Project-Active-green)

## 📌 Overview

**Coastal Safety AI** is an intelligent computer vision system designed
to improve beach safety by detecting dangerous situations in real time.

The system analyzes live video feeds using **YOLOv8 object detection**
and **centroid tracking** to monitor swimmers, identify risky behavior,
and detect hazardous zones such as rip currents, deep water areas, and
rocky shorelines.

By generating real-time alerts and automated reports, the system assists
lifeguards in responding faster to potential drowning incidents.

------------------------------------------------------------------------

## 🚀 Key Features

-   Real-time **swimmer detection using YOLOv8**
-   **Rip current risk detection** based on movement patterns
-   **Deep water monitoring** to identify possible drowning situations
-   **Rock proximity alerts**
-   **Crowd density monitoring** in wave-impact zones
-   **Real-time alert snapshots**
-   **Multichannel emergency alerts**
    -   SMS
    -   WhatsApp
    -   Telegram
    -   Email
-   **Voice alert system using Text-to-Speech**
-   **Alert heatmap visualization**
-   **Automatic PDF safety report generation**
-   **Multilingual interface**

------------------------------------------------------------------------

## 🧠 How the System Works

1.  Video feed is captured from **webcam or uploaded video**.
2.  **YOLOv8 model** detects objects such as:
    -   Person
    -   Boat
    -   Surfboard
3.  **Centroid tracking** tracks movement of detected individuals.
4.  Risk analysis identifies dangerous scenarios including:
    -   Rip current movement
    -   Stationary person in deep water
    -   Swimmer near rocks
    -   High crowd density in wave zones
5.  System generates **alerts, snapshots, and logs**.

------------------------------------------------------------------------

## 🏗 System Workflow

Video Input → YOLOv8 Detection → Centroid Tracking → Risk Detection →
Alert Generation → Dashboard & Reports

------------------------------------------------------------------------

## 🖥️ Technologies Used

-   Python
-   YOLOv8 (Ultralytics)
-   OpenCV
-   Streamlit
-   NumPy & Pandas
-   Matplotlib
-   gTTS (Text-to-Speech)
-   Twilio API

------------------------------------------------------------------------

## 📂 Project Structure

Coastal-Safety-AI │ ├── coastal_safety_app.py ├── requirements.txt ├──
sample_videos ├── outputs │ ├── logs │ ├── snapshots │ └── reports └──
README.md

------------------------------------------------------------------------

## ⚙ Installation

### Clone the repository

git clone https://github.com/sriharshini2022/Coastal-Safety-AI.git

### Install dependencies

pip install -r requirements.txt

### Run the application

streamlit run coastal_safety_app.py

------------------------------------------------------------------------

## 📊 Outputs Generated

-   Alert Snapshots
-   Event Logs (CSV)
-   Risk Heatmaps
-   PDF Safety Reports

------------------------------------------------------------------------

## 🌍 Impact

This project supports **AI-driven coastal monitoring systems** that help
reduce drowning incidents and improve beach safety.

Aligned with: - **SDG 14 -- Life Below Water** - **SDG 13 -- Climate
Action**

------------------------------------------------------------------------

