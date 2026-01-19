# OpenAI Realtime Vision + Audio MVP

A Python application that captures webcam video and microphone audio, sends them to OpenAI's Realtime API, and plays back audio responses.

## Features

- **Webcam capture**: Sends frames at configurable intervals (default: 1 second)
- **Microphone input**: Continuous audio streaming to the model
- **Audio output**: Plays model's spoken responses through speakers
- **Async architecture**: Non-blocking I/O using asyncio

## Requirements

- Python 3.10+
- OpenAI API key with Realtime API access
- Webcam
- Microphone
- Speakers/headphones

## Installation

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the main application:
```bash
python main.py
```

### Command Line Options

- `--mode`: Input mode - `camera`, `screen`, or `none` (default: `camera`)
- `--interval`: Frame capture interval in seconds (default: `1.0`)

Examples:
```bash
# Camera mode with 2-second frame intervals
python main.py --mode camera --interval 2.0

# Screen capture mode
python main.py --mode screen

# Audio-only mode (no video)
python main.py --mode none
```

### Controls

- Type text messages and press Enter to send
- Type `q` and press Enter to quit

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Python Application                     │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐    │
│  │  Webcam  │  │   Mic    │  │  Audio Playback    │    │
│  │ (OpenCV) │  │(PyAudio) │  │    (PyAudio)       │    │
│  └────┬─────┘  └────┬─────┘  └─────────▲──────────┘    │
│       │             │                   │               │
│       ▼             ▼                   │               │
│  ┌──────────────────────────────────────────────────┐  │
│  │           Async Event Loop (asyncio)              │  │
│  └────────────────────┬─────────────────────────────┘  │
└───────────────────────┼─────────────────────────────────┘
                        │ WebSocket
                        ▼
             ┌─────────────────────┐
             │  OpenAI Realtime    │
             │  API (gpt-4o)       │
             └─────────────────────┘
```

## Troubleshooting

### PyAudio Installation Issues

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Ubuntu/Debian:**
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

**Windows:**
```bash
pip install pyaudio
```

### Camera Access

Ensure your terminal/IDE has camera access permissions in System Preferences (macOS) or equivalent.

## License

MIT
