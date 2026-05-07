# Project Architecture 🏗️

This project follows a modular AI pipeline architecture:

1. **Input Layer:** Captures video stream using OpenCV.
2. **Preprocessing Layer:** Uses MediaPipe to identify hand landmarks and normalize coordinates.
3. **Inference Layer:** A TensorFlow Lite model (optimized for speed) classifies the gesture.
4. **Action Layer:** Triggers Text-to-Speech or updates the React.js dashboard.

### Component Diagram
`Webcam -> OpenCV -> MediaPipe -> TensorFlow -> Speech Synthesis`
