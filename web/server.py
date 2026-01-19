"""
FastAPI server for OpenAI Realtime API ephemeral token generation.
Serves the web frontend and handles token requests.
"""

import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

SESSION_CONFIG = {
    "type": "realtime",
    "model": "gpt-realtime",
    "instructions": """You are a helpful AI assistant with vision capabilities.
You can see what the user's camera shows and hear what they say.
- Be concise and natural in your responses
- Describe what you see when relevant
- Respond conversationally to audio input
- If you see something interesting, mention it proactively""",
    "audio": {
        "input": {
            "format": {
                "type": "audio/pcm",
                "rate": 24000
            }
        },
        "output": {
            "format": {
                "type": "audio/pcm",
                "rate": 24000
            },
            "voice": "coral"
        }
    }
}


@app.get("/token")
async def get_token():
    """Generate an ephemeral token for WebRTC connection."""
    if not OPENAI_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"error": "OPENAI_API_KEY not configured"}
        )
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.openai.com/v1/realtime/client_secrets",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={"session": SESSION_CONFIG},
                timeout=30.0
            )
            
            if response.status_code != 200:
                return JSONResponse(
                    status_code=response.status_code,
                    content={"error": response.text}
                )
            
            return response.json()
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )


@app.post("/session")
async def create_session(request: Request):
    """Create a session using the unified interface (alternative approach)."""
    if not OPENAI_API_KEY:
        return PlainTextResponse(
            status_code=500,
            content="OPENAI_API_KEY not configured"
        )
    
    sdp = await request.body()
    
    async with httpx.AsyncClient() as client:
        try:
            from httpx import Response
            import json
            
            # Create multipart form data
            files = {
                "sdp": ("sdp", sdp, "application/sdp"),
                "session": ("session", json.dumps(SESSION_CONFIG), "application/json")
            }
            
            response = await client.post(
                "https://api.openai.com/v1/realtime/calls",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                },
                files=files,
                timeout=30.0
            )
            
            if response.status_code != 200 and response.status_code != 201:
                return PlainTextResponse(
                    status_code=response.status_code,
                    content=response.text
                )
            
            return PlainTextResponse(content=response.text)
        except Exception as e:
            return PlainTextResponse(
                status_code=500,
                content=str(e)
            )


# Serve static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    import socket
    
    # Get local IP for mobile access
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("\n" + "="*60)
    print("OpenAI Realtime Vision Web Server")
    print("="*60)
    print(f"\nAccess from this computer: http://localhost:3000")
    print(f"Access from iPhone/mobile: http://{local_ip}:3000")
    print("\nNote: Camera/mic access requires HTTPS on mobile browsers.")
    print("For local network, you may need to use a tunnel or self-signed cert.")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=3000)
