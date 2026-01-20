"""
FastAPI server for OpenAI Realtime API ephemeral token generation.
Serves the web frontend and handles token requests.
"""

import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")
load_dotenv()  # Also check current directory

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-realtime")

SYSTEM_INSTRUCTIONS = """You are Tutora, a helpful AI tutor with vision capabilities.
You can see what the user's camera shows and hear what they say.
ALWAYS respond ONLY to what you see in the LATEST image frame provided to you.
- Be concise and natural in your responses
- Respond conversationally to audio input
- DO NOT repeat things (unless explicitly asked by the user)"""

SESSION_CONFIG = {
    "type": "realtime",
    "model": MODEL,
    "instructions": SYSTEM_INSTRUCTIONS
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
                print(f"Token API error: {response.status_code} - {response.text}")
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
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


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
