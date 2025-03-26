from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="Orpheus Browser",
    description="Browser interface for Orpheus TTS",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Define available voices and default voice
AVAILABLE_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
DEFAULT_VOICE = "tara"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "tts.html",
        {
            "request": request,
            "voices": AVAILABLE_VOICES,
            "DEFAULT_VOICE": DEFAULT_VOICE,
            "text": "",
            "error": None,
            "success": None,
            "generation_time": None
        }
    )

if __name__ == "__main__":
    print("🌐 Starting Orpheus Browser on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000) 