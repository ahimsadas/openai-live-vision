# OpenAI Realtime Vision

A browser-based client using WebRTC for real-time audio/video streaming with OpenAI's Realtime API.

## Features

- **WebRTC audio** with echo cancellation, noise suppression, and auto gain control
- **Camera preview** with front/back camera switching (mobile)
- **Real-time transcription** of both user and AI speech
- **Mute/camera toggle** controls

## Requirements

- Python 3.10+
- OpenAI API key with Realtime API access
- Modern browser with WebRTC support
- Webcam and Microphone

## Setup

### 1. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Create Virtual Environment

```bash
cd web
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd web
python run_https.py
```

Open `https://localhost:3000` in your browser (accept the self-signed certificate warning).

For mobile access, use `https://YOUR_IP:3000` (the IP will be printed when the server starts).

### Why HTTPS?

Browsers require HTTPS for `getUserMedia()` access to camera/microphone (except localhost on some browsers). The server generates a self-signed certificate automatically.

## Project Structure

```
openai-live-vision/
├── .env.example         # Environment template
├── README.md
└── web/
    ├── server.py        # FastAPI server with token endpoint
    ├── run_https.py     # HTTPS runner with self-signed cert
    ├── requirements.txt # Dependencies
    └── static/
        ├── index.html   # Web UI
        └── app.js       # Client-side JavaScript
```

## Notes

- The Realtime API uses PCM audio at 24kHz sample rate
- Images are sent as base64 JPEG via WebRTC data channel
- Uses server-side VAD for turn detection
