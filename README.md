# Gesture-to-Speech Translator 🖐️

Communication should be for everyone. I'm building this project to bridge the gap between Sign Language and spoken word using AI.

### What it does:
The app uses a webcam to track hand gestures in real-time. Using a custom TensorFlow model, it recognizes specific signs and translates them into text and audio.

### The Tech:
- **AI/ML:** TensorFlow & MediaPipe (for hand tracking).
- **Computer Vision:** OpenCV for the camera feed.
- **Frontend:** A clean dashboard built with React.js.


*Note: This project is currently under active development!*

---

## 🚀 Development Progress

### Log 1: Getting the "Eyes" Working
I've officially started building the system! I decided to begin with **MediaPipe** because it provides a rock-solid foundation for hand tracking without me needing to train a massive model from scratch on day one.

**What I did:**
- I set up a Python virtual environment to keep my dependencies clean.
- I wrote `vision_test.py` to bridge my webcam feed with the MediaPipe AI.
- I implemented real-time landmark detection (21 points per hand).

**How it works:**
The script captures each frame from my camera, converts it to RGB, and passes it to the MediaPipe hands model. The model then returns a list of coordinates for my finger joints, which I'm drawing back onto the screen using OpenCV. This is the foundation I'll use later to collect data and train my custom gesture classifier.

### Log 2: Stability and Upgrades
I hit a major roadblock with Python 3.14—it was too new for the standard AI libraries and caused the system to crash. Instead of giving up, I performed a "technical pivot."

**What I did:**
- I downgraded my environment to **Python 3.12 (Stable)** to ensure long-term reliability.
- I upgraded my code to use the **Modern MediaPipe Tasks API** instead of the legacy solutions.
- Since the new API doesn't have an automatic "drawer," I manually programmed the skeleton logic using OpenCV.

**Result:**
The vision engine is now rock-solid. It tracks hands faster than before and the manual drawing logic gives me full control over how the AI "sees" the world. I'm now ready to start teaching it actual gestures.

