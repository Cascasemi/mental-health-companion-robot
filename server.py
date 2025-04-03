import os
import numpy as np
import whisper
import requests
from fastapi import FastAPI, UploadFile, HTTPException, status
from pymongo import MongoClient
from dotenv import load_dotenv
import uvicorn
from datetime import datetime
from typing import Dict, Any

# Load environment variables
load_dotenv()

app = FastAPI()

# --- Initialize Services ---
try:
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"), connectTimeoutMS=5000)
    db = client["mental_health_db"]
    users = db["users"]
    client.admin.command('ping')  # Test connection
    print("✅ Successfully connected to MongoDB")
except Exception as e:
    print(f"❌ MongoDB connection error: {e}")
    raise

try:
    whisper_model = whisper.load_model("base")
    print("✅ Whisper model loaded")
except Exception as e:
    print(f"❌ Whisper loading error: {e}")
    raise


# Health check endpoints
@app.get("/")
async def root():
    return {"message": "Mental Health Companion API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "ready", "timestamp": datetime.now().isoformat()}


@app.get("/test_connection")
async def test_connection():
    return {"message": "Connection test successful", "success": True}


# Core functionality
def get_llama_response(prompt: str) -> str:
    """Get response from Llama API"""
    api_key = os.getenv("LLAMA_API_KEY")
    if not api_key:
        raise ValueError("Llama API key not configured")

    model = os.getenv("LLAMA_MODEL", "llama-3-70b-instruct")
    api_base = os.getenv("LLAMA_API_BASE", "https://api.llama.ai/v1")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a compassionate mental health assistant. Respond with brief, empathetic answers (1-2 sentences)."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }

    try:
        response = requests.post(
            f"{api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Llama API Connection Error: {e}")
        return "I'm having connection issues. Please try again later."
    except Exception as e:
        print(f"Llama API Processing Error: {e}")
        return "I'm having trouble formulating a response."


@app.post("/register_user")
async def register_user(user_data: Dict[str, Any]):
    """Register a new user"""
    try:
        if "name" not in user_data or not user_data["name"].strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Name is required"
            )

        result = users.insert_one({
            "name": user_data["name"].strip(),
            "conversations": [],
            "created_at": datetime.now(),
            "last_active": datetime.now()
        })
        return {
            "status": "success",
            "user_id": str(result.inserted_id),
            "name": user_data["name"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/process_audio")
async def process_audio(file: UploadFile):
    try:
        # Validate file
        if not file.content_type or not file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only audio files are accepted"
            )

        # 1. Convert speech to text
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty audio file received"
            )

        audio_np = np.frombuffer(audio_bytes, np.int16)
        if len(audio_np) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid audio data"
            )

        audio_float = audio_np.astype(np.float32) / 32768.0
        result = whisper_model.transcribe(audio_float)
        user_text = result.get("text", "").strip()

        if not user_text:
            return {"text": "I didn't catch that. Could you repeat?"}

        print(f"User: {user_text}")

        # 2. Get AI response
        ai_text = get_llama_response(user_text)
        print(f"AI: {ai_text}")

        # 3. Store conversation
        users.update_one(
            {"name": "current_user"},
            {"$push": {"conversations": {
                "user_text": user_text,
                "ai_response": ai_text,
                "timestamp": datetime.now()
            }}},
            upsert=True
        )

        return {"text": ai_text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing error: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=os.getenv("SERVER_HOST", "127.0.0.1"),
        port=int(os.getenv("SERVER_PORT", "8000")),
        reload=True
    )