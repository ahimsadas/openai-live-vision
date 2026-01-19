# OpenAI Realtime Vision MVP

A Python application that captures webcam video and microphone audio, sends them to OpenAI's Realtime API, and plays back the AI's audio responses.

## Features

- **Webcam capture**: Sends frames at configurable intervals (default: 1 second)
- **Microphone input**: Streams audio to the API in real-time
- **Audio playback**: Plays AI responses through your speakers
- **Async architecture**: Non-blocking I/O using asyncio

## Requirements

- Python 3.10+
- OpenAI API key with Realtime API access
- Webcam
- Microphone
- Speakers/Headphones

## Setup

1. Create and activate virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create `.env` file with your API key:
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

## Usage

Run the application:
```bash
python main.py
```

Options:
```bash
python main.py --mode camera    # Webcam mode (default)
python main.py --mode screen    # Screen capture mode
python main.py --mode none      # Audio only, no video
python main.py --interval 2.0   # Send frames every 2 seconds
```

Type messages in the console to send text input. Type `q` to quit.

## Web App

A browser-based client using WebRTC for real-time audio/video streaming.

### Setup

```bash
cd web
pip install -r requirements.txt
```

### Running

```bash
python run_https.py
```

Open `https://localhost:3000` in your browser (accept the self-signed certificate warning).
Correct port will be printed in CLI.

### Features

- **WebRTC audio** with echo cancellation, noise suppression, and auto gain control
- **Camera preview** with front/back camera switching (mobile)
- **Real-time transcription** of both user and AI speech
- **Mute/camera toggle** controls

### Why HTTPS?

Browsers require HTTPS for `getUserMedia()` access to camera/microphone (except localhost on some browsers). The server generates a self-signed certificate automatically.

## Architecture

```
Webcam → Frame Queue → 
                        → Output Queue → WebSocket → OpenAI Realtime API
Microphone → Audio Queue →                              ↓
                                              Audio Response Queue → Speaker
```

## Notes

- The Realtime API uses PCM audio at 24kHz sample rate
- Images are resized to max 1024px and sent as base64 JPEG
- Uses semantic VAD for turn detection
