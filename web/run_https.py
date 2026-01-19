"""
Run the web server with HTTPS for mobile device access.
Generates a self-signed certificate if needed.
"""

import os
import subprocess
import sys

CERT_FILE = "cert.pem"
KEY_FILE = "key.pem"

def generate_self_signed_cert():
    """Generate a self-signed certificate for local HTTPS."""
    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        print("Using existing certificates...")
        return
    
    print("Generating self-signed certificate...")
    
    # Generate self-signed cert using openssl
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:4096",
        "-keyout", KEY_FILE, "-out", CERT_FILE,
        "-days", "365", "-nodes",
        "-subj", "/CN=localhost",
        "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:192.168.1.33"
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print("Certificate generated successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error generating certificate: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("OpenSSL not found. Please install it or use ngrok instead.")
        sys.exit(1)


def main():
    import socket
    
    # Change to the web directory (where server.py and static/ are located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Get local IP
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "192.168.x.x"
    
    # Try to get the actual network IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        pass
    
    generate_self_signed_cert()
    
    print("\n" + "="*60)
    print("OpenAI Realtime Vision Web Server (HTTPS)")
    print("="*60)
    print(f"\nAccess from this computer: https://localhost:3000")
    print(f"Access from iPhone/mobile: https://{local_ip}:3000")
    print("\n⚠️  On iPhone, you'll see a security warning.")
    print("   Tap 'Advanced' > 'Proceed to site' to continue.")
    print("="*60 + "\n")
    
    # Run uvicorn with SSL
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=3000,
        ssl_keyfile=KEY_FILE,
        ssl_certfile=CERT_FILE,
        reload=False
    )


if __name__ == "__main__":
    main()
