"""
OpenAI Realtime Vision MVP

Captures webcam video and microphone audio, sends to OpenAI Realtime API,
and plays back audio responses.

Usage:
    python main.py --mode camera    # Webcam mode (default)
    python main.py --mode screen    # Screen capture mode
    python main.py --mode none      # Audio only
    python main.py --interval 2.0   # Frame interval in seconds
"""

import os
import sys
import select
import asyncio
import base64
import io
import json
import traceback
import argparse

import cv2
import pyaudio
import PIL.Image
import numpy as np
import websockets
from dotenv import load_dotenv

load_dotenv()

# ANSI color codes for prettier terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    
    BG_CYAN = "\033[46m"
    BG_GREEN = "\033[42m"


def log_info(msg: str):
    print(f"{Colors.CYAN}ℹ{Colors.RESET}  {msg}")

def log_success(msg: str):
    print(f"{Colors.GREEN}✓{Colors.RESET}  {msg}")

def log_warning(msg: str):
    print(f"{Colors.YELLOW}⚠{Colors.RESET}  {msg}")

def log_error(msg: str):
    print(f"{Colors.RED}✗{Colors.RESET}  {msg}")

def log_event(icon: str, msg: str):
    print(f"{Colors.MAGENTA}{icon}{Colors.RESET}  {msg}")


# Audio configuration - OpenAI Realtime API uses 24kHz PCM16
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 24000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# OpenAI Realtime API endpoint
REALTIME_API_URL = "wss://api.openai.com/v1/realtime"
MODEL = "gpt-realtime"

DEFAULT_MODE = "camera"
DEFAULT_FRAME_INTERVAL = 1.0

# Frame capture intervals (from .env or defaults)
IDLE_FRAME_INTERVAL = float(os.environ.get("IDLE_FRAME_INTERVAL_MS", 6000)) / 1000.0  # 6 seconds
SPEAK_FRAME_INTERVAL = float(os.environ.get("SPEAK_FRAME_INTERVAL_MS", 500)) / 1000.0  # 0.5s = 2fps

# System instructions for the AI
SYSTEM_INSTRUCTIONS = """You are a helpful AI assistant with vision capabilities. 
You can see what the user's camera shows and hear what they say.
Respond naturally and conversationally to what you see and hear.
Be concise but helpful. Describe what you see when asked.
If you notice something interesting or relevant, feel free to mention it."""


class RealtimeVisionClient:
    def __init__(self, video_mode: str = DEFAULT_MODE, frame_interval: float = DEFAULT_FRAME_INTERVAL):
        self.video_mode = video_mode
        self.frame_interval = frame_interval
        
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.audio_in_queue = None  # Queue for received audio to play
        self.out_queue = None       # Queue for outgoing messages (audio/images)
        
        self.websocket = None
        self.audio_stream = None
        self.pya = pyaudio.PyAudio()
        
        self.running = True
        self.is_responding = False  # Track when AI is streaming a response
        self.is_user_speaking = False  # Track when user is speaking (for frame rate switching)
        self.frame_interval_event = asyncio.Event()  # Signal to update frame interval

    async def connect(self):
        """Connect to OpenAI Realtime API via WebSocket."""
        url = f"{REALTIME_API_URL}?model={MODEL}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        log_info("Connecting to OpenAI Realtime API...")
        self.websocket = await websockets.connect(
            url,
            additional_headers=headers,
            max_size=None  # No limit on message size
        )
        log_success("Connected!")
        
        # Wait for session.created event
        response = await self.websocket.recv()
        event = json.loads(response)
        if event.get("type") == "session.created":
            session_id = event.get('session', {}).get('id', 'unknown')[:12]
            log_success(f"Session created: {Colors.DIM}{session_id}...{Colors.RESET}")
        
        # Configure the session
        await self.configure_session()

    async def configure_session(self):
        """Send session.update to configure the realtime session."""
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": SYSTEM_INSTRUCTIONS,
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 500,
                    "silence_duration_ms": 500
                }
            }
        }
        
        await self.websocket.send(json.dumps(session_config))
        log_success("Session configured with voice and vision capabilities")

    async def send_text(self):
        """Handle text input from console with non-blocking stdin."""
        print(f"{Colors.BLUE}>{Colors.RESET} ", end="", flush=True)
        buffer = ""
        
        while self.running:
            # Check if stdin has data (with short timeout to allow checking self.running)
            ready, _, _ = await asyncio.to_thread(
                select.select, [sys.stdin], [], [], 0.2
            )
            
            if not self.running:
                break
            
            if ready:
                line = sys.stdin.readline()
                if not line:  # EOF
                    break
                text = line.strip()
                
                if text.lower() == "q":
                    log_info("Quit requested...")
                    self.running = False
                    break
                    
                if text:
                    await self.send_text_message(text)
                
                if self.running:
                    print(f"{Colors.BLUE}>{Colors.RESET} ", end="", flush=True)

    async def send_text_message(self, text: str):
        """Send a text message to the API."""
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text
                    }
                ]
            }
        }
        await self.websocket.send(json.dumps(event))
        
        # Request a response
        await self.websocket.send(json.dumps({"type": "response.create"}))

    def _capture_and_encode_frame(self, cap) -> tuple | None:
        """Capture and encode a frame from the webcam. Returns (raw_frame, encoded_data) or None."""
        ret, frame = cap.read()
        if not ret:
            return None
        
        # Convert BGR to RGB for encoding
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        
        # Resize to max 512px while maintaining aspect ratio (reduces token usage)
        img.thumbnail([512, 512])
        
        # Encode as JPEG with lower quality (reduces token usage)
        image_io = io.BytesIO()
        img.save(image_io, format="JPEG", quality=70)
        image_io.seek(0)
        
        image_bytes = image_io.read()
        encoded_data = {
            "type": "image",
            "mime_type": "image/jpeg",
            "data": base64.standard_b64encode(image_bytes).decode()
        }
        
        # Return both raw frame (for preview) and encoded data (for API)
        return (frame, encoded_data)

    def _get_current_frame_interval(self) -> float:
        """Get current frame interval based on speaking state."""
        if self.is_user_speaking:
            return SPEAK_FRAME_INTERVAL
        return IDLE_FRAME_INTERVAL

    async def _interruptible_sleep(self, duration: float):
        """Sleep that can be interrupted when frame interval changes."""
        try:
            # Wait for either timeout or event signal
            await asyncio.wait_for(self.frame_interval_event.wait(), timeout=duration)
            # Event was set - clear it and return early
            self.frame_interval_event.clear()
        except asyncio.TimeoutError:
            # Normal timeout, continue
            pass

    async def capture_frames(self):
        """Continuously capture frames from webcam with dynamic interval switching."""
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        
        if not cap.isOpened():
            log_error("Could not open webcam - frame sending disabled")
            return
        
        log_success(f"Webcam opened (idle: {IDLE_FRAME_INTERVAL}s, speak: {SPEAK_FRAME_INTERVAL}s)")
        log_info("Camera preview window opened. Type 'q' + Enter to quit.")
        
        try:
            while self.running:
                result = await asyncio.to_thread(self._capture_and_encode_frame, cap)
                if result is None:
                    await asyncio.sleep(0.1)
                    continue
                
                raw_frame, encoded_data = result
                
                # Show preview on main thread (required for macOS)
                cv2.imshow("Camera Preview", raw_frame)
                cv2.waitKey(1)
                
                await self.out_queue.put(encoded_data)
                
                # Use interruptible sleep with current interval
                await self._interruptible_sleep(self._get_current_frame_interval())
        finally:
            cap.release()
            cv2.destroyAllWindows()
            log_info("Webcam released")

    def _get_screen(self) -> dict | None:
        """Capture and encode a screenshot."""
        try:
            import mss
            sct = mss.mss()
            monitor = sct.monitors[0]
            screenshot = sct.grab(monitor)
            
            # Convert to PIL Image
            img = PIL.Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            img.thumbnail([512, 512])
            
            # Encode as JPEG with lower quality (reduces token usage)
            image_io = io.BytesIO()
            img.save(image_io, format="JPEG", quality=70)
            image_io.seek(0)
            
            image_bytes = image_io.read()
            return {
                "type": "image",
                "mime_type": "image/jpeg",
                "data": base64.standard_b64encode(image_bytes).decode()
            }
        except ImportError:
            log_error("mss not installed. Run: pip install mss")
            return None

    async def capture_screen(self):
        """Continuously capture screen with dynamic interval switching."""
        log_success(f"Screen capture started (idle: {IDLE_FRAME_INTERVAL}s, speak: {SPEAK_FRAME_INTERVAL}s)")
        
        while self.running:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                await asyncio.sleep(1.0)
                continue
            
            await self.out_queue.put(frame)
            
            # Use interruptible sleep with current interval
            await self._interruptible_sleep(self._get_current_frame_interval())

    async def capture_audio(self):
        """Capture audio from microphone and send to API."""
        mic_info = self.pya.get_default_input_device_info()
        log_success(f"Microphone: {Colors.DIM}{mic_info['name']}{Colors.RESET}")
        
        self.audio_stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=int(mic_info["index"]),
            frames_per_buffer=CHUNK_SIZE,
        )
        
        try:
            while self.running:
                data = await asyncio.to_thread(
                    self.audio_stream.read, 
                    CHUNK_SIZE, 
                    exception_on_overflow=False
                )
                audio_msg = {
                    "type": "audio",
                    "data": base64.b64encode(data).decode()
                }
                await self.out_queue.put(audio_msg)
        finally:
            if self.audio_stream:
                self.audio_stream.close()

    async def send_to_api(self):
        """Send queued messages (audio/images) to the API."""
        while self.running:
            try:
                msg = await asyncio.wait_for(self.out_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            
            if msg["type"] == "audio":
                # Send audio as input_audio_buffer.append
                event = {
                    "type": "input_audio_buffer.append",
                    "audio": msg["data"]
                }
                await self.websocket.send(json.dumps(event))
                
            elif msg["type"] == "image":
                # Send image as conversation item with data URL format
                data_url = f"data:{msg['mime_type']};base64,{msg['data']}"
                event = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": data_url
                            }
                        ]
                    }
                }
                await self.websocket.send(json.dumps(event))
                if not self.is_responding:
                    print(f"{Colors.DIM}📷 Frame sent{Colors.RESET}")

    async def receive_from_api(self):
        """Receive events from the API and handle them."""
        try:
            async for message in self.websocket:
                if not self.running:
                    break
                    
                event = json.loads(message)
                event_type = event.get("type", "")
                
                if event_type == "response.audio.delta":
                    # Received audio chunk - decode and queue for playback
                    audio_data = base64.b64decode(event.get("delta", ""))
                    await self.audio_in_queue.put(audio_data)
                    
                elif event_type == "response.audio_transcript.delta":
                    # Print transcript as it comes in
                    self.is_responding = True
                    transcript = event.get("delta", "")
                    print(f"{Colors.GREEN}{transcript}{Colors.RESET}", end="", flush=True)
                    
                elif event_type == "response.audio_transcript.done":
                    self.is_responding = False
                    print()  # Newline after transcript
                    
                elif event_type == "response.audio.done":
                    self.is_responding = False
                    
                elif event_type == "input_audio_buffer.speech_started":
                    # Interruption: end current response line cleanly, then show listening
                    if self.is_responding:
                        print()  # Newline to end partial response
                        self.is_responding = False
                    log_event("🎤", f"{Colors.YELLOW}Listening...{Colors.RESET}")
                    # Switch to fast frame capture mode
                    self.is_user_speaking = True
                    self.frame_interval_event.set()  # Interrupt current sleep to switch immediately
                    # Clear any pending audio playback when user starts speaking (interruption)
                    while not self.audio_in_queue.empty():
                        try:
                            self.audio_in_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    
                elif event_type == "input_audio_buffer.speech_stopped":
                    log_event("🎤", f"{Colors.DIM}Processing...{Colors.RESET}")
                    # Switch back to idle frame capture mode
                    self.is_user_speaking = False
                    self.frame_interval_event.set()  # Interrupt current sleep to switch immediately
                    
                elif event_type == "response.done":
                    response = event.get("response", {})
                    if response.get("status") == "failed":
                        error = response.get("status_details", {}).get("error", {})
                        log_error(f"Response failed: {error.get('message', 'Unknown error')}")
                        
                elif event_type == "error":
                    error = event.get("error", {})
                    log_error(error.get('message', 'Unknown error'))
                    
                elif event_type == "session.updated":
                    log_success("Session updated")
                    
        except websockets.exceptions.ConnectionClosed as e:
            log_warning(f"Connection closed: {e}")
            self.running = False

    async def play_audio(self):
        """Play received audio through speakers."""
        output_stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        
        try:
            while self.running:
                try:
                    audio_data = await asyncio.wait_for(
                        self.audio_in_queue.get(), 
                        timeout=0.1
                    )
                    await asyncio.to_thread(output_stream.write, audio_data)
                except asyncio.TimeoutError:
                    continue
        finally:
            output_stream.close()

    async def run(self):
        """Main run loop."""
        tasks = []
        try:
            await self.connect()
            
            self.audio_in_queue = asyncio.Queue()
            self.out_queue = asyncio.Queue(maxsize=10)
            
            # Create all tasks
            tasks.append(asyncio.create_task(self.send_text(), name="send_text"))
            tasks.append(asyncio.create_task(self.capture_audio(), name="capture_audio"))
            tasks.append(asyncio.create_task(self.play_audio(), name="play_audio"))
            tasks.append(asyncio.create_task(self.send_to_api(), name="send_to_api"))
            tasks.append(asyncio.create_task(self.receive_from_api(), name="receive_from_api"))
            
            if self.video_mode == "camera":
                tasks.append(asyncio.create_task(self.capture_frames(), name="capture_frames"))
            elif self.video_mode == "screen":
                tasks.append(asyncio.create_task(self.capture_screen(), name="capture_screen"))
            
            # Wait for any task to complete (usually send_text when user quits)
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            # Cancel all pending tasks
            self.running = False
            for task in pending:
                task.cancel()
            
            # Wait for cancellation to complete (with timeout)
            if pending:
                await asyncio.wait(pending, timeout=2.0)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log_error(f"{e}")
            traceback.print_exc()
        finally:
            print()
            log_info("Shutting down...")
            self.running = False
            
            # Cancel any remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # Close resources
            if self.websocket:
                try:
                    await self.websocket.close()
                except:
                    pass
            if self.audio_stream:
                try:
                    self.audio_stream.stop_stream()
                    self.audio_stream.close()
                except:
                    pass
            # Suppress CoreAudio errors during PyAudio cleanup (redirect fd-level stderr)
            devnull = os.open(os.devnull, os.O_WRONLY)
            old_stderr = os.dup(2)
            os.dup2(devnull, 2)
            try:
                self.pya.terminate()
            except:
                pass
            os.dup2(old_stderr, 2)
            os.close(devnull)
            os.close(old_stderr)
            
            log_success("Cleanup complete")
            
            # Force exit to kill any lingering threads from asyncio.to_thread
            os._exit(0)


def main():
    parser = argparse.ArgumentParser(description="OpenAI Realtime Vision MVP")
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        choices=["camera", "screen", "none"],
        help="Video input mode: camera, screen, or none (audio only)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_FRAME_INTERVAL,
        help="Interval between frame captures in seconds (default: 1.0)"
    )
    args = parser.parse_args()
    
    print()
    print(f"{Colors.BOLD}🎥 OpenAI Realtime Vision{Colors.RESET}")
    print(f"{Colors.DIM}Mode: {args.mode} | Interval: {args.interval}s | Quit: 'q' + Enter{Colors.RESET}")
    print()
    
    client = RealtimeVisionClient(
        video_mode=args.mode,
        frame_interval=args.interval
    )
    asyncio.run(client.run())


if __name__ == "__main__":
    main()
