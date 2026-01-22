/**
 * OpenAI Realtime Vision - WebRTC Client
 * Connects to OpenAI Realtime API via WebRTC with camera and microphone
 */

// Frame capture intervals (configurable)
const IDLE_FRAME_INTERVAL_MS = 6000;   // 1 frame every 6 seconds when idle
const SPEAK_FRAME_INTERVAL_MS = 500;   // 2 frames per second when user speaking

class RealtimeClient {
    constructor() {
        this.pc = null;
        this.dc = null;
        this.localStream = null;
        this.audioElement = null;
        this.isConnected = false;
        this.isMuted = false;
        this.isCameraOff = false;
        this.frameInterval = null;
        this.facingMode = 'user'; // 'user' = front, 'environment' = back
        this.isResponding = false;  // Track when AI is streaming a response
        this.currentTranscriptEl = null;  // Current transcript element for live updates
        this.isUserSpeaking = false;  // Track when user is speaking (for frame rate switching)
        this.cameraAvailable = false;  // Track camera availability
        this.availableCameras = [];
        this.currentCameraIndex = 0;
        this.debugMode = false;  // Debug mode to show captured frames
        
        // DOM elements
        this.videoEl = document.getElementById('localVideo');
        this.videoContainer = document.querySelector('.video-container');
        this.btnConnect = document.getElementById('btnConnect');
        this.btnMute = document.getElementById('btnMute');
        this.btnCamera = document.getElementById('btnCamera');
        this.btnSwitch = document.getElementById('btnSwitch');
        this.btnDebug = document.getElementById('btnDebug');
        this.debugCanvas = document.getElementById('debugCanvas');
        this.warmupOverlay = document.getElementById('warmupOverlay');
        this.statusDot = document.getElementById('statusDot');
        this.statusText = document.getElementById('statusText');
        this.transcriptContent = document.getElementById('transcriptContent');
        this.errorContainer = document.getElementById('errorContainer');
        
        this.setupEventListeners();
        this.setupVideoAspectListeners();
        this.enumerateCameras();
    }
    
    isMobileDevice() {
        return /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
    }
    
    async enumerateCameras() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            this.availableCameras = devices.filter(d => d.kind === 'videoinput');
            console.log('Available cameras:', this.availableCameras.length, 'Mobile:', this.isMobileDevice());
            
            // Only show switch button on mobile devices with 2+ cameras
            // facingMode (front/back) switching is only meaningful on mobile
            if (this.isMobileDevice() && this.availableCameras.length >= 2) {
                this.btnSwitch.style.display = '';
            } else {
                this.btnSwitch.style.display = 'none';
            }
        } catch (e) {
            console.log('Could not enumerate cameras');
            // Hide switch button on error
            this.btnSwitch.style.display = 'none';
        }
    }
    
    setupEventListeners() {
        this.btnConnect.addEventListener('click', () => this.toggleConnection());
        this.btnMute.addEventListener('click', () => this.toggleMute());
        this.btnCamera.addEventListener('click', () => this.toggleCamera());
        this.btnSwitch.addEventListener('click', () => this.switchCamera());
        this.btnDebug.addEventListener('click', () => this.toggleDebug());
    }
    
    toggleDebug() {
        this.debugMode = !this.debugMode;
        this.btnDebug.classList.toggle('active', this.debugMode);
        this.debugCanvas.classList.toggle('visible', this.debugMode);
        console.log('Debug mode:', this.debugMode ? 'ON' : 'OFF');
    }

    setupVideoAspectListeners() {
        if (!this.videoEl) return;
        this.videoEl.addEventListener('loadedmetadata', () => this.scheduleVideoAspectUpdate());
        this.videoEl.addEventListener('resize', () => this.scheduleVideoAspectUpdate());
    }
    
    async toggleConnection() {
        if (this.isConnected) {
            await this.disconnect();
        } else {
            await this.connect();
        }
    }
    
    async connect() {
        try {
            // Show warmup immediately when user taps call
            this.warmupOverlay.classList.remove('hidden');
            this.showStatus('Warming up...', false);
            this.hideError();
            
            // Get user media (camera + microphone)
            this.localStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: this.facingMode, width: { ideal: 640 }, height: { ideal: 480 } },
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                }
            });
            
            // Re-enumerate cameras after permission granted (iOS Safari needs this)
            await this.enumerateCameras();
            
            // Update video mirroring based on camera
            this.updateVideoMirror();
            
            this.videoEl.srcObject = this.localStream;
            this.scheduleVideoAspectUpdate(this.localStream.getVideoTracks()[0]);
            
            // Get ephemeral token from our server
            this.showStatus('Getting token...', false);
            const tokenResponse = await fetch('/token');
            const tokenData = await tokenResponse.json();
            
            if (tokenData.error) {
                throw new Error(tokenData.error);
            }
            
            const ephemeralKey = tokenData.value;
            
            // Create peer connection
            this.pc = new RTCPeerConnection({
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
            });
            
            // Set up audio playback from the model
            this.audioElement = document.createElement('audio');
            this.audioElement.autoplay = true;
            this.pc.ontrack = (e) => {
                this.audioElement.srcObject = e.streams[0];
            };
            
            // Add audio track
            const audioTrack = this.localStream.getAudioTracks()[0];
            if (audioTrack) {
                this.pc.addTrack(audioTrack, this.localStream);
            }
            
            // Set up data channel for events
            this.dc = this.pc.createDataChannel('oai-events');
            this.dc.onopen = async () => {
                console.log('Data channel opened');
                
                // Model warmup: 5 seconds AFTER connection to let model stabilize
                console.log('[Client] Connection established, starting 5s model warmup...');
                await new Promise(resolve => setTimeout(resolve, 5000));
                
                this.warmupOverlay.classList.add('hidden');
                console.log('[Client] Warmup complete, enabling video');
                this.showStatus('Connected', true);
                
                this.startSendingFrames();
            };
            this.dc.onmessage = (e) => this.handleServerEvent(JSON.parse(e.data));
            this.dc.onerror = (e) => console.error('Data channel error:', e);
            
            // Create and send offer
            this.showStatus('Connecting to OpenAI...', false);
            const offer = await this.pc.createOffer();
            await this.pc.setLocalDescription(offer);
            
            const sdpResponse = await fetch('https://api.openai.com/v1/realtime/calls', {
                method: 'POST',
                body: offer.sdp,
                headers: {
                    'Authorization': `Bearer ${ephemeralKey}`,
                    'Content-Type': 'application/sdp'
                }
            });
            
            if (!sdpResponse.ok) {
                const errorText = await sdpResponse.text();
                throw new Error(`SDP error: ${errorText}`);
            }
            
            const answerSdp = await sdpResponse.text();
            await this.pc.setRemoteDescription({ type: 'answer', sdp: answerSdp });
            
            this.isConnected = true;
            this.showStatus('Connected', true);
            this.btnConnect.classList.remove('btn-connect');
            this.btnConnect.classList.add('btn-disconnect');
            
            this.addTranscript('system', 'Connected! Speak or show something to the camera.');
            
        } catch (error) {
            console.error('Connection error:', error);
            this.showError(error.message);
            this.showStatus('Error', false);
            await this.disconnect();
        }
    }
    
    async disconnect() {
        this.stopSendingFrames();
        
        if (this.dc) {
            this.dc.close();
            this.dc = null;
        }
        
        if (this.pc) {
            this.pc.close();
            this.pc = null;
        }
        
        if (this.localStream) {
            this.localStream.getTracks().forEach(track => track.stop());
            this.localStream = null;
        }
        
        if (this.audioElement) {
            this.audioElement.srcObject = null;
            this.audioElement = null;
        }
        
        this.videoEl.srcObject = null;
        this.isConnected = false;
        this.isUserSpeaking = false;  // Reset speaking state
        this.cameraAvailable = false;  // Reset camera state
        this.showStatus('Disconnected', false);
        this.btnConnect.classList.remove('btn-disconnect');
        this.btnConnect.classList.add('btn-connect');
    }
    
    toggleMute() {
        if (!this.localStream) return;
        
        const audioTrack = this.localStream.getAudioTracks()[0];
        if (audioTrack) {
            this.isMuted = !this.isMuted;
            audioTrack.enabled = !this.isMuted;
            this.btnMute.classList.toggle('muted', this.isMuted);
        }
    }
    
    toggleCamera() {
        if (!this.localStream) return;
        
        const videoTrack = this.localStream.getVideoTracks()[0];
        if (videoTrack) {
            this.isCameraOff = !this.isCameraOff;
            videoTrack.enabled = !this.isCameraOff;
            this.btnCamera.classList.toggle('off', this.isCameraOff);
            
            if (this.isCameraOff) {
                this.stopSendingFrames();
            } else {
                this.startSendingFrames();
                this.scheduleVideoAspectUpdate(videoTrack);
            }
        }
    }
    
    async switchCamera() {
        if (!this.localStream) return;
        
        // Toggle facing mode
        this.facingMode = this.facingMode === 'user' ? 'environment' : 'user';
        
        try {
            // Stop current video track
            const oldVideoTrack = this.localStream.getVideoTracks()[0];
            if (oldVideoTrack) {
                oldVideoTrack.stop();
            }
            
            // Get new video stream with different camera
            const newStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: this.facingMode, width: { ideal: 640 }, height: { ideal: 480 } },
                audio: false
            });
            
            const newVideoTrack = newStream.getVideoTracks()[0];
            
            // Replace track in local stream
            this.localStream.removeTrack(oldVideoTrack);
            this.localStream.addTrack(newVideoTrack);
            
            // Update video element
            this.videoEl.srcObject = this.localStream;
            this.updateVideoMirror();
            this.scheduleVideoAspectUpdate(newVideoTrack);
            
            console.log('Switched to:', this.facingMode === 'user' ? 'front camera' : 'back camera');
        } catch (e) {
            console.error('Error switching camera:', e);
            // Revert facing mode on error
            this.facingMode = this.facingMode === 'user' ? 'environment' : 'user';
        }
    }
    
    updateVideoMirror() {
        // No mirroring - show camera as-is
        this.videoEl.style.transform = 'scaleX(1)';
    }

    scheduleVideoAspectUpdate(videoTrack = this.localStream?.getVideoTracks?.()[0]) {
        this.updateVideoAspectRatio(videoTrack);
        if (!this.videoEl) return;
        const applyUpdate = () => this.updateVideoAspectRatio(videoTrack);
        if (typeof this.videoEl.requestVideoFrameCallback === 'function') {
            this.videoEl.requestVideoFrameCallback(() => applyUpdate());
        } else {
            setTimeout(applyUpdate, 150);
        }
    }

    updateVideoAspectRatio(videoTrack = this.localStream?.getVideoTracks?.()[0]) {
        if (!this.videoContainer) return;
        const settings = videoTrack && videoTrack.getSettings ? videoTrack.getSettings() : {};
        const width = this.videoEl.videoWidth || settings.width;
        const height = this.videoEl.videoHeight || settings.height;
        if (width && height) {
            this.videoContainer.style.setProperty('--video-aspect', `${width} / ${height}`);
        }
    }
    
    getCurrentFrameInterval() {
        return this.isUserSpeaking ? SPEAK_FRAME_INTERVAL_MS : IDLE_FRAME_INTERVAL_MS;
    }
    
    startSendingFrames() {
        // Check if camera is available
        if (!this.localStream) {
            this.cameraAvailable = false;
            console.log('Camera unavailable - frame sending disabled');
            return;
        }
        
        const videoTrack = this.localStream.getVideoTracks()[0];
        if (!videoTrack || !videoTrack.enabled) {
            this.cameraAvailable = false;
            console.log('Video track unavailable - frame sending disabled');
            return;
        }
        
        this.cameraAvailable = true;
        
        // Clear any existing interval to prevent duplicates
        this.stopSendingFrames();
        
        const interval = this.getCurrentFrameInterval();
        console.log(`Starting frame capture: ${interval}ms interval (${this.isUserSpeaking ? 'speak' : 'idle'} mode)`);
        
        this.frameInterval = setInterval(() => {
            if (this.isConnected && !this.isCameraOff && this.dc?.readyState === 'open') {
                this.captureAndSendFrame();
            }
        }, interval);
        
        // Send first frame immediately
        setTimeout(() => this.captureAndSendFrame(), 100);
    }
    
    stopSendingFrames() {
        if (this.frameInterval) {
            clearInterval(this.frameInterval);
            this.frameInterval = null;
        }
    }
    
    switchFrameMode(speaking) {
        // Only switch if state actually changed
        if (this.isUserSpeaking === speaking) return;
        
        this.isUserSpeaking = speaking;
        
        // Only restart if camera is available and connected
        if (this.cameraAvailable && this.isConnected && !this.isCameraOff) {
            this.startSendingFrames();  // This will clear old interval and start new one
        }
    }
    
    captureAndSendFrame() {
        if (!this.localStream || !this.dc || this.dc.readyState !== 'open') return;
        
        const videoTrack = this.localStream.getVideoTracks()[0];
        if (!videoTrack || !videoTrack.enabled) return;
        
        // Create canvas to capture frame
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        // Get original dimensions
        let width = this.videoEl.videoWidth || 640;
        let height = this.videoEl.videoHeight || 480;
        
        // Resize to max 512px while maintaining aspect ratio (reduces token usage)
        const maxSize = 512;
        if (width > maxSize || height > maxSize) {
            if (width > height) {
                height = Math.round(height * (maxSize / width));
                width = maxSize;
            } else {
                width = Math.round(width * (maxSize / height));
                height = maxSize;
            }
        }
        
        canvas.width = width;
        canvas.height = height;
        
        // Draw current frame (no flip)
        ctx.drawImage(this.videoEl, 0, 0, canvas.width, canvas.height);
        
        // Convert to base64 JPEG data URL with lower quality (reduces token usage)
        const dataUrl = canvas.toDataURL('image/jpeg', 0.70);
        
        // Send as conversation item with image_url format
        const event = {
            type: 'conversation.item.create',
            item: {
                type: 'message',
                role: 'user',
                content: [
                    {
                        type: 'input_image',
                        image_url: dataUrl
                    }
                ]
            }
        };
        
        // If debug mode is on, draw the captured frame to the debug canvas
        if (this.debugMode) {
            this.debugCanvas.width = canvas.width;
            this.debugCanvas.height = canvas.height;
            const debugCtx = this.debugCanvas.getContext('2d');
            debugCtx.drawImage(canvas, 0, 0);
        }
        
        try {
            this.dc.send(JSON.stringify(event));
            console.log('Frame sent');
        } catch (e) {
            console.error('Error sending frame:', e);
        }
    }
    
    handleServerEvent(event) {
        console.log('Server event:', event.type, event);
        
        switch (event.type) {
            case 'session.created':
                console.log('Session created:', event.session?.id);
                break;
                
            case 'session.updated':
                console.log('Session updated');
                this.addTranscript('system', 'Session updated');
                break;
                
            case 'input_audio_buffer.speech_started':
                // Interruption: user started speaking
                if (this.isResponding) {
                    // End current response cleanly
                    this.finishCurrentTranscript();
                    this.isResponding = false;
                }
                this.showStatus('Listening...', true);
                console.log('🎤 Listening...');
                // Switch to fast frame capture mode
                this.switchFrameMode(true);
                // CRITICAL: Send current frame immediately so model has latest visual context
                // This prevents the "stale frame" issue where model responds to old images
                this.captureAndSendFrame();
                break;
                
            case 'input_audio_buffer.speech_stopped':
                this.showStatus('Processing...', true);
                console.log('🎤 Processing...');
                // Switch back to idle frame capture mode
                this.switchFrameMode(false);
                break;
                
            case 'response.audio_transcript.delta':
            case 'response.output_audio_transcript.delta':
                // Real-time transcript of AI response - show live as it streams
                this.isResponding = true;
                if (event.delta) {
                    this.appendToCurrentTranscript(event.delta);
                }
                break;
                
            case 'response.audio_transcript.done':
            case 'response.output_audio_transcript.done':
                this.isResponding = false;
                this.finishCurrentTranscript();
                this.showStatus('Connected', true);
                break;
                
            case 'response.audio.done':
            case 'response.output_audio.done':
                this.isResponding = false;
                break;
                
            case 'response.text.done':
                if (event.text) {
                    this.addTranscript('assistant', event.text);
                }
                break;
                
            case 'response.done':
                // Check for errors in response
                const response = event.response;
                if (response?.status === 'failed') {
                    const error = response.status_details?.error;
                    console.error('Response failed:', error);
                    this.showError(error?.message || 'Response failed');
                }
                this.isResponding = false;
                break;
                
            case 'error':
                console.error('API Error:', event.error);
                this.showError(event.error?.message || 'Unknown error');
                break;
        }
    }
    
    appendToCurrentTranscript(delta) {
        // Create or append to current transcript element for live streaming
        if (!this.currentTranscriptEl) {
            this.currentTranscriptEl = document.createElement('div');
            this.currentTranscriptEl.className = 'transcript-item assistant';
            this.currentTranscriptEl.textContent = '🤖 AI: ';
            
            // Remove placeholder if exists
            const placeholder = this.transcriptContent.querySelector('[style]');
            if (placeholder) placeholder.remove();
            
            // Prepend at top so latest is always visible
            this.transcriptContent.prepend(this.currentTranscriptEl);
        }
        
        this.currentTranscriptEl.textContent += delta;
    }
    
    finishCurrentTranscript() {
        // Finalize the current streaming transcript
        this.currentTranscriptEl = null;
    }
    
    showStatus(text, connected) {
        this.statusText.textContent = text;
        this.statusDot.classList.toggle('connected', connected);
    }
    
    showError(message) {
        this.errorContainer.textContent = message;
        this.errorContainer.classList.remove('hidden');
    }
    
    hideError() {
        this.errorContainer.classList.add('hidden');
    }
    
    addTranscript(role, text) {
        const item = document.createElement('div');
        item.className = `transcript-item ${role}`;
        
        const prefix = role === 'user' ? '🗣️ You: ' : 
                       role === 'assistant' ? '🤖 AI: ' : '📌 ';
        item.textContent = prefix + text;
        
        // Remove placeholder if exists
        const placeholder = this.transcriptContent.querySelector('[style]');
        if (placeholder) placeholder.remove();
        
        // Prepend at top so latest is always visible
        this.transcriptContent.prepend(item);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('OpenAI Realtime Vision v2 - Debug mode available');
    window.realtimeClient = new RealtimeClient();
});
