# Orpheus-FASTAPI by Lex-au
# https://github.com/Lex-au/Orpheus-FastAPI
# Description: Main FastAPI server for Orpheus Text-to-Speech

import os
import time
import asyncio
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv

# Function to ensure .env file exists
def ensure_env_file_exists():
    """Create a default .env file if one doesn't exist"""
    if not os.path.exists(".env") and os.path.exists(".env.example"):
        try:
            # Copy .env.example to .env
            with open(".env.example", "r") as example_file:
                with open(".env", "w") as env_file:
                    env_file.write(example_file.read())
            print("✅ Created default configuration file at .env")
        except Exception as e:
            print(f"⚠️ Error creating default .env file: {e}")

# Ensure .env file exists before loading environment variables
ensure_env_file_exists()

# Load environment variables from .env file
load_dotenv(override=True)

from fastapi import FastAPI, Request, Form, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
import uuid

from tts_engine import generate_speech_from_api, AVAILABLE_VOICES, DEFAULT_VOICE, generate_tokens_from_api, tokens_decoder

# Create FastAPI app
app = FastAPI(
    title="Orpheus-FASTAPI",
    description="High-performance Text-to-Speech server using Orpheus-FASTAPI",
    version="1.0.0"
)

# We'll use FastAPI's built-in startup complete mechanism
# The log message "INFO:     Application startup complete." indicates
# that the application is ready

# Ensure directories exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount directories for serving files
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# API models
class SpeechRequest(BaseModel):
    input: str
    model: str = "orpheus"
    voice: str = DEFAULT_VOICE
    response_format: str = "wav"
    speed: float = 1.0

class APIResponse(BaseModel):
    status: str
    voice: str
    output_file: str
    generation_time: float

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

# WebSocket endpoint for streaming audio
@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            # Generate a unique context ID for this audio stream
            context_id = str(uuid.uuid4())
            
            # Send start signal with context ID
            await websocket.send_json({
                'type': 'start',
                'context_id': context_id
            })
            
            # Extract parameters from request
            prompt = request_data.get('prompt', '')
            voice = request_data.get('voice', DEFAULT_VOICE)
            temperature = request_data.get('temperature', 0.6)
            top_p = request_data.get('top_p', 0.9)
            max_tokens = request_data.get('max_tokens', 8192)
            
            if not prompt:
                await websocket.send_json({
                    'type': 'error',
                    'context_id': context_id,
                    'error': 'Missing prompt'
                })
                continue
            
            # Generate tokens and convert to audio chunks
            token_gen = generate_tokens_from_api(
                prompt=prompt,
                voice=voice,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )
            
            # Process tokens and send audio chunks
            async for audio_chunk in tokens_decoder(token_gen):
                if audio_chunk:
                    await websocket.send_json({
                        'type': 'audio_chunk',
                        'context_id': context_id,
                        'chunk': audio_chunk.hex()
                    })
            
            # Send end signal
            await websocket.send_json({
                'type': 'generation_complete',
                'context_id': context_id
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await websocket.send_json({
            'type': 'error',
            'context_id': context_id if 'context_id' in locals() else None,
            'error': str(e)
        })
        manager.disconnect(websocket)

# Streaming endpoint for HTTP requests
@app.post("/v1/audio/speech/stream")
async def stream_speech(request: SpeechRequest):
    """
    Stream audio generation using OpenAI-compatible streaming response.
    """
    if not request.input:
        raise HTTPException(status_code=400, detail="Missing input text")
    
    async def generate():
        token_gen = generate_tokens_from_api(
            prompt=request.input,
            voice=request.voice,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=MAX_TOKENS
        )
        
        async for audio_chunk in tokens_decoder(token_gen):
            if audio_chunk:
                yield audio_chunk
    
    return StreamingResponse(
        generate(),
        media_type="audio/wav",
        headers={
            "Content-Type": "audio/wav",
            "Transfer-Encoding": "chunked"
        }
    )

# OpenAI-compatible API endpoint
@app.post("/v1/audio/speech")
async def create_speech_api(request: SpeechRequest):
    """
    Generate speech from text using the Orpheus TTS model.
    Compatible with OpenAI's /v1/audio/speech endpoint.
    
    For longer texts (>1000 characters), batched generation is used
    to improve reliability and avoid truncation issues.
    """
    if not request.input:
        raise HTTPException(status_code=400, detail="Missing input text")
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/{request.voice}_{timestamp}.wav"
    
    # Check if we should use batched generation
    use_batching = len(request.input) > 1000
    if use_batching:
        print(f"Using batched generation for long text ({len(request.input)} characters)")
    
    # Generate speech with automatic batching for long texts
    start = time.time()
    generate_speech_from_api(
        prompt=request.input,
        voice=request.voice,
        output_file=output_path,
        use_batching=use_batching,
        max_batch_chars=1000  # Process in ~1000 character chunks (roughly 1 paragraph)
    )
    end = time.time()
    generation_time = round(end - start, 2)
    
    # Return audio file
    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename=f"{request.voice}_{timestamp}.wav"
    )

# Legacy API endpoint for compatibility
@app.post("/speak")
async def speak(request: Request):
    """Legacy endpoint for compatibility with existing clients"""
    data = await request.json()
    text = data.get("text", "")
    voice = data.get("voice", DEFAULT_VOICE)

    if not text:
        return JSONResponse(
            status_code=400, 
            content={"error": "Missing 'text'"}
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/{voice}_{timestamp}.wav"
    
    # Check if we should use batched generation for longer texts
    use_batching = len(text) > 1000
    if use_batching:
        print(f"Using batched generation for long text ({len(text)} characters)")
    
    # Generate speech with batching for longer texts
    start = time.time()
    generate_speech_from_api(
        prompt=text, 
        voice=voice, 
        output_file=output_path,
        use_batching=use_batching,
        max_batch_chars=1000
    )
    end = time.time()
    generation_time = round(end - start, 2)

    return JSONResponse(content={
        "status": "ok",
        "voice": voice,
        "output_file": output_path,
        "generation_time": generation_time
    })

# Web UI routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Redirect to web UI"""
    return templates.TemplateResponse(
        "tts.html",
        {"request": request, "voices": AVAILABLE_VOICES}
    )

@app.get("/web/", response_class=HTMLResponse)
async def web_ui(request: Request):
    """Main web UI for TTS generation"""
    # Get current config for the Web UI
    config = get_current_config()
    return templates.TemplateResponse(
        "tts.html",
        {"request": request, "voices": AVAILABLE_VOICES, "config": config}
    )

@app.get("/get_config")
async def get_config():
    """Get current configuration from .env file or defaults"""
    config = get_current_config()
    return JSONResponse(content=config)

@app.post("/save_config")
async def save_config(request: Request):
    """Save configuration to .env file"""
    data = await request.json()
    
    # Convert values to proper types
    for key, value in data.items():
        if key in ["ORPHEUS_MAX_TOKENS", "ORPHEUS_API_TIMEOUT", "ORPHEUS_PORT", "ORPHEUS_SAMPLE_RATE"]:
            try:
                data[key] = str(int(value))
            except (ValueError, TypeError):
                pass
        elif key in ["ORPHEUS_TEMPERATURE", "ORPHEUS_TOP_P"]:  # Removed ORPHEUS_REPETITION_PENALTY since it's hardcoded now
            try:
                data[key] = str(float(value))
            except (ValueError, TypeError):
                pass
    
    # Write configuration to .env file
    with open(".env", "w") as f:
        for key, value in data.items():
            f.write(f"{key}={value}\n")
    
    return JSONResponse(content={"status": "ok", "message": "Configuration saved successfully. Restart server to apply changes."})

@app.post("/restart_server")
async def restart_server():
    """Restart the server by touching a file that triggers Uvicorn's reload"""
    import threading
    
    def touch_restart_file():
        # Wait a moment to let the response get back to the client
        time.sleep(0.5)
        
        # Create or update restart.flag file to trigger reload
        restart_file = "restart.flag"
        with open(restart_file, "w") as f:
            f.write(str(time.time()))
            
        print("🔄 Restart flag created, server will reload momentarily...")
    
    # Start the touch operation in a separate thread
    threading.Thread(target=touch_restart_file, daemon=True).start()
    
    # Return success response
    return JSONResponse(content={"status": "ok", "message": "Server is restarting. Please wait a moment..."})

def get_current_config():
    """Read current configuration from .env.example and .env files"""
    # Default config from .env.example
    default_config = {}
    if os.path.exists(".env.example"):
        with open(".env.example", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    default_config[key] = value
    
    # Current config from .env
    current_config = {}
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    current_config[key] = value
    
    # Merge configs, with current taking precedence
    config = {**default_config, **current_config}
    
    # Add current environment variables
    for key in config:
        env_value = os.environ.get(key)
        if env_value is not None:
            config[key] = env_value
    
    return config

@app.post("/web/", response_class=HTMLResponse)
async def generate_from_web(
    request: Request,
    text: str = Form(...),
    voice: str = Form(DEFAULT_VOICE)
):
    """Handle form submission from web UI"""
    if not text:
        return templates.TemplateResponse(
            "tts.html",
            {
                "request": request,
                "error": "Please enter some text.",
                "voices": AVAILABLE_VOICES
            }
        )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/{voice}_{timestamp}.wav"
    
    # Check if we should use batched generation for longer texts
    use_batching = len(text) > 1000
    if use_batching:
        print(f"Using batched generation for long text from web form ({len(text)} characters)")
    
    # Generate speech with batching for longer texts
    start = time.time()
    generate_speech_from_api(
        prompt=text, 
        voice=voice, 
        output_file=output_path,
        use_batching=use_batching,
        max_batch_chars=1000
    )
    end = time.time()
    generation_time = round(end - start, 2)
    
    return templates.TemplateResponse(
        "tts.html",
        {
            "request": request,
            "success": True,
            "text": text,
            "voice": voice,
            "output_file": output_path,
            "generation_time": generation_time,
            "voices": AVAILABLE_VOICES
        }
    )

@app.get("/stream", response_class=HTMLResponse)
async def stream_demo(request: Request):
    """Streaming demo page"""
    return templates.TemplateResponse(
        "stream.html",
        {"request": request, "voices": AVAILABLE_VOICES}
    )

if __name__ == "__main__":
    import uvicorn
    
    # Check for required settings
    required_settings = ["ORPHEUS_HOST", "ORPHEUS_PORT"]
    missing_settings = [s for s in required_settings if s not in os.environ]
    if missing_settings:
        print(f"⚠️ Missing environment variable(s): {', '.join(missing_settings)}")
        print("   Using fallback values for server startup.")
    
    # Get host and port from environment variables with better error handling
    try:
        host = os.environ.get("ORPHEUS_HOST")
        if not host:
            print("⚠️ ORPHEUS_HOST not set, using 0.0.0.0 as fallback")
            host = "0.0.0.0"
    except Exception:
        print("⚠️ Error reading ORPHEUS_HOST, using 0.0.0.0 as fallback")
        host = "0.0.0.0"
        
    try:
        port = int(os.environ.get("ORPHEUS_PORT", "5005"))
    except (ValueError, TypeError):
        print("⚠️ Invalid ORPHEUS_PORT value, using 5005 as fallback")
        port = 5005
    
    print(f"🔥 Starting Orpheus-FASTAPI Server on {host}:{port}")
    print(f"💬 Web UI available at http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
    print(f"📖 API docs available at http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs")
    
    # Read current API_URL for user information
    api_url = os.environ.get("ORPHEUS_API_URL")
    if not api_url:
        print("⚠️ ORPHEUS_API_URL not set. Please configure in .env file before generating speech.")
    else:
        print(f"🔗 Using LLM inference server at: {api_url}")
        
    # Include restart.flag in the reload_dirs to monitor it for changes
    extra_files = ["restart.flag"] if os.path.exists("restart.flag") else []
    
    # Start with reload enabled to allow automatic restart when restart.flag changes
    uvicorn.run("app:app", host=host, port=port, reload=True, reload_dirs=["."], reload_includes=["*.py", "*.html", "restart.flag"])
