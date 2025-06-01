# Face-Attendance-System
# 🎯 Face Recognition Attendance System

A real-time face recognition attendance system built using Python. It uses the webcam to detect and recognize faces of known individuals and automatically logs their attendance with a timestamp in a CSV file.

---

## 📸 Demo

> The webcam opens, detects faces, and logs recognized individuals as present.
>  
> *(Add a screenshot or demo GIF here if needed)*

---

## 📂 Project Structure


---

## ✅ Features

- Real-time face detection via webcam
- Matches faces against known persons
- Logs attendance with timestamp
- Saves attendance data to `attendance_log.csv`
- Simple, lightweight, and fast

---

## 🧰 Tools & Libraries Used

- **Python 3.11** — Programming language
- **OpenCV** — Webcam access and frame display
- **face_recognition** — Face detection and matching
- **dlib** — Backend for face encoding
- **NumPy** — Array operations and calculations
- **Pandas** — Manage and export attendance logs
- **CMake** — Required to build dlib on Windows

---

## 💻 System Requirements

- Windows OS (Tested on Windows 10/11)
- Python 3.8–3.11
- Webcam (internal or USB)

---

## ⚙️ Installation & Setup (Windows)

### 1. Clone the Repository

```powershell
git clone https://github.com/rk-005/face-attendance.git
cd face-attendance

