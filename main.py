"""
OpenAI Realtime Vision + Audio MVP

Captures webcam video and microphone audio, sends to OpenAI Realtime API,
and plays back audio responses.

Usage:
    python main.py --mode camera --interval 1.0
"""

import os
import asyncio
import base64
import io
import json
import argparse
import traceback

import cv2
import pyaudio
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import websockets

load_dotenv()

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 24000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "gpt-realtime"
WEBSOCKET_URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"

DEFAULT_MODE = "camera"
DEFAULT_INTERVAL = 1.0

SYSTEM_INSTRUCTIONS = """You are a helpful AI assistant with vision capabilities. 
You can see what the user's camera shows and hear what they say.
- Be concise and natural in your responses
- Describe what you see when relevant
- Respond conversationally to audio input
- If you see something interesting or relevant, mention it proactively
"""

pya = pyaudio.PyAudio()


class RealtimeSession:
    def __init__(self, video_mode=DEFAULT_MODE, frame_interval=DEFAULT_INTERVAL):
        self.video_mode = video_mode
        self.frame_interval = frame_interval
        
        self.audio_in_queue = None
        self.out_queue = None
        self.ws = None
        self.audio_stream = None
        self.running = True

    async def connect(self):
        """Establish WebSocket connection to OpenAI Realtime API."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        print("Connecting to OpenAI Realtime API...")
        self.ws = await websockets.connect(
            WEBSOCKET_URL,
            additional_headers=headers,
            max_size=None
        )
        print("Connected!")
        
        session_created = await self.ws.recv()
        data = json.loads(session_created)
        if data.get("type") == "session.created":
            print(f"Session created: {data.get('session', {}).get('id', 'unknown')}")
        
        await self.configure_session()

    async def configure_session(self):
        """Configure the realtime session with audio settings and instructions."""
        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": SYSTEM_INSTRUCTIONS,
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                }
            }
        }
        
        await self.ws.send(json.dumps(session_update))
        print("Session configured with audio and vision capabilities")

    async def send_text(self):
        """Handle text input from the user."""
        while self.running:
            try:
                text = await asyncio.to_thread(input, "message > ")
                if text.lower() == "q":
                    self.running = False
                    break
                if text.strip():
                    event = {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": text}
                            ]
                        }
                    }
                    await self.ws.send(json.dumps(event))
                    await self.ws.send(json.dumps({"type": "response.create"}))
            except Exception as e:
                if self.running:
                    print(f"Text input error: {e}")
                break

    def _get_frame(self, cap):
        """Capture and process a single frame from the webcam."""
        ret, frame = cap.read()
        if not ret:
            return None, None
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail([512, 512])
        
        image_io = io.BytesIO()
        img.save(image_io, format="JPEG", quality=70)
        image_io.seek(0)
        
        image_bytes = image_io.read()
        return base64.b64encode(image_bytes).decode(), frame

    async def get_frames(self):
        """Continuously capture frames from the webcam at specified intervals."""
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print(f"Webcam opened. Sending frames every {self.frame_interval} seconds...")
        print("Video preview window opened - Press 'q' in the window to quit")
        
        try:
            while self.running:
                frame_b64, frame = await asyncio.to_thread(self._get_frame, cap)
                if frame_b64 is None:
                    break
                
                if frame is not None:
                    preview_frame = frame.copy()
                    cv2.putText(preview_frame, "Sending to AI...", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Webcam Preview", preview_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.running = False
                        break
                
                await self.out_queue.put({
                    "type": "image",
                    "data": frame_b64
                })
                
                await asyncio.sleep(self.frame_interval)
        finally:
            cv2.destroyAllWindows()
            cap.release()
            print("Webcam released")

    def _get_screen(self):
        """Capture the screen."""
        try:
            import mss
            sct = mss.mss()
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)
            
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            img.thumbnail([512, 512])
            
            image_io = io.BytesIO()
            img.save(image_io, format="JPEG", quality=70)
            image_io.seek(0)
            
            return base64.b64encode(image_io.read()).decode()
        except ImportError:
            print("Screen capture requires 'mss' package: pip install mss")
            return None

    async def get_screen(self):
        """Continuously capture screen at specified intervals."""
        print(f"Screen capture mode. Sending frames every {self.frame_interval} seconds...")
        
        while self.running:
            frame_b64 = await asyncio.to_thread(self._get_screen)
            if frame_b64 is None:
                break
            
            await self.out_queue.put({
                "type": "image",
                "data": frame_b64
            })
            
            await asyncio.sleep(self.frame_interval)

    async def send_media(self):
        """Send queued media (images and audio) to the API."""
        while self.running:
            try:
                msg = await asyncio.wait_for(self.out_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            
            try:
                if msg["type"] == "image":
                    event = {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{msg['data']}"
                                }
                            ]
                        }
                    }
                    await self.ws.send(json.dumps(event))
                    
                elif msg["type"] == "audio":
                    event = {
                        "type": "input_audio_buffer.append",
                        "audio": msg["data"]
                    }
                    await self.ws.send(json.dumps(event))
                    
            except Exception as e:
                if self.running:
                    print(f"Send error: {e}")

    async def listen_audio(self):
        """Capture audio from the microphone and queue it for sending."""
        try:
            mic_info = pya.get_default_input_device_info()
            print(f"Using microphone: {mic_info['name']}")
        except Exception as e:
            print(f"Error getting microphone: {e}")
            return
        
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=int(mic_info["index"]),
            frames_per_buffer=CHUNK_SIZE,
        )
        
        print("Microphone active. Speak to interact...")
        
        try:
            while self.running:
                try:
                    data = await asyncio.to_thread(
                        self.audio_stream.read, 
                        CHUNK_SIZE, 
                        exception_on_overflow=False
                    )
                    audio_b64 = base64.b64encode(data).decode()
                    await self.out_queue.put({
                        "type": "audio",
                        "data": audio_b64
                    })
                except Exception as e:
                    if self.running:
                        print(f"Audio capture error: {e}")
                    break
        finally:
            if self.audio_stream:
                self.audio_stream.close()

    async def receive_messages(self):
        """Receive and process messages from the API."""
        while self.running:
            try:
                message = await self.ws.recv()
                data = json.loads(message)
                event_type = data.get("type", "")
                
                if event_type == "response.audio.delta":
                    audio_data = data.get("delta", "")
                    if audio_data:
                        audio_bytes = base64.b64decode(audio_data)
                        await self.audio_in_queue.put(audio_bytes)
                
                elif event_type == "response.audio_transcript.delta":
                    transcript = data.get("delta", "")
                    if transcript:
                        print(transcript, end="", flush=True)
                
                elif event_type == "response.audio_transcript.done":
                    print()
                
                elif event_type == "input_audio_buffer.speech_started":
                    while not self.audio_in_queue.empty():
                        try:
                            self.audio_in_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                
                elif event_type == "error":
                    error = data.get("error", {})
                    print(f"\nAPI Error: {error.get('message', 'Unknown error')}")
                
                elif event_type == "session.updated":
                    print("Session settings updated")
                    
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
                self.running = False
                break
            except Exception as e:
                if self.running:
                    print(f"Receive error: {e}")

    async def play_audio(self):
        """Play received audio through speakers."""
        stream = await asyncio.to_thread(
            pya.open,
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
                    await asyncio.to_thread(stream.write, audio_data)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"Playback error: {e}")
        finally:
            stream.close()

    async def run(self):
        """Main run loop."""
        try:
            await self.connect()
            
            self.audio_in_queue = asyncio.Queue()
            self.out_queue = asyncio.Queue(maxsize=10)
            
            tasks = [
                asyncio.create_task(self.send_text()),
                asyncio.create_task(self.send_media()),
                asyncio.create_task(self.listen_audio()),
                asyncio.create_task(self.receive_messages()),
                asyncio.create_task(self.play_audio()),
            ]
            
            if self.video_mode == "camera":
                tasks.append(asyncio.create_task(self.get_frames()))
            elif self.video_mode == "screen":
                tasks.append(asyncio.create_task(self.get_screen()))
            
            print("\n" + "="*50)
            print("OpenAI Realtime Vision + Audio MVP")
            print("="*50)
            print("- Speak naturally to interact")
            print("- Type messages and press Enter")
            print("- Type 'q' to quit")
            print("="*50 + "\n")
            
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            self.running = False
            
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
        finally:
            if self.audio_stream:
                self.audio_stream.close()
            if self.ws:
                await self.ws.close()
            pya.terminate()
            print("Session ended")


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI Realtime Vision + Audio MVP"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        choices=["camera", "screen", "none"],
        help="Video input mode (default: camera)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help="Frame capture interval in seconds (default: 1.0)"
    )
    args = parser.parse_args()
    
    session = RealtimeSession(
        video_mode=args.mode,
        frame_interval=args.interval
    )
    asyncio.run(session.run())


if __name__ == "__main__":
    main()
