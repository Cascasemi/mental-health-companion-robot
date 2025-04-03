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
    client = MongoClient(
        os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        connectTimeoutMS=5000,
        serverSelectionTimeoutMS=5000
    )
    db = client["mental_health_db"]
    users = db["users"]
    # Immediate connection test
    client.admin.command('ping')
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


# --- Health Endpoints ---
@app.get("/")
async def root():
    return {
        "service": "Mental Health Companion",
        "status": "operational",
        "llm_provider": "Groq",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "mongodb": "connected",
            "whisper": "loaded",
            "groq_api": "configured" if os.getenv("GROQ_API_KEY") else "missing_key"
        }
    }


# --- Core API Functions ---
def get_groq_response(prompt: str) -> str:
    """Get response from Groq's ultra-fast API"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Groq API key not configured in environment variables")

    model = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
    api_base = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "MentalHealthCompanion/1.0"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": """You are a compassionate mental health assistant. 
                Respond with brief (1-2 sentence), empathetic answers.
                Maintain a warm, non-judgmental tone.
                If user seems in crisis, suggest professional help."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 150,
        "top_p": 0.9,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.2
    }

    try:
        response = requests.post(
            f"{api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=10  # Groq typically responds in <500ms
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        error_msg = f"Groq API Error: {str(e)}"
        if hasattr(e, 'response') and e.response:
            error_msg += f" | Response: {e.response.text[:200]}"
        print(error_msg)
        return "I'm having trouble connecting to my assistance system. Please try again shortly."
    except Exception as e:
        print(f"Unexpected Groq processing error: {str(e)}")
        return "I'm having difficulty formulating a response right now."


# --- User Management ---
@app.post("/register_user")
async def register_user(user_data: Dict[str, Any]):
    """Register a new user with the system"""
    try:
        if not user_data.get("name", "").strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Name must be provided and non-empty"
            )

        user_doc = {
            "name": user_data["name"].strip(),
            "conversations": [],
            "created_at": datetime.now(),
            "last_active": datetime.now(),
            "settings": {
                "preferred_voice": "default",
                "emergency_contact": None
            }
        }

        result = users.insert_one(user_doc)

        return {
            "status": "success",
            "user_id": str(result.inserted_id),
            "name": user_doc["name"],
            "timestamp": user_doc["created_at"].isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"User registration failed: {str(e)}"
        )


# --- Audio Processing ---
@app.post("/process_audio")
async def process_audio(file: UploadFile, user_id: str = None):
    """Process audio input and return AI response"""
    try:
        # 1. Validate input
        if not file.content_type or not file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only audio files are accepted (WAV, MP3, etc.)"
            )

        # 2. Process audio
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Received empty audio file"
            )

        audio_np = np.frombuffer(audio_bytes, np.int16)
        if len(audio_np) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid audio data format"
            )

        # 3. Convert speech to text
        audio_float = audio_np.astype(np.float32) / 32768.0
        result = whisper_model.transcribe(audio_float)
        user_text = result.get("text", "").strip()

        if not user_text:
            return {"text": "I didn't catch that. Could you please repeat?"}

        print(f"User: {user_text}")

        # 4. Get AI response
        ai_text = get_groq_response(user_text)
        print(f"AI: {ai_text}")

        # 5. Store conversation
        conversation_entry = {
            "user_text": user_text,
            "ai_response": ai_text,
            "timestamp": datetime.now(),
            "audio_length": len(audio_np) / 16000  # duration in seconds
        }

        update_filter = {"_id": user_id} if user_id else {"name": "current_user"}
        users.update_one(
            update_filter,
            {
                "$push": {"conversations": conversation_entry},
                "$set": {"last_active": datetime.now()}
            },
            upsert=True
        )

        return {"text": ai_text}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio processing failed: {str(e)}"
        )


# --- Server Startup ---
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=os.getenv("SERVER_HOST", "127.0.0.1"),
        port=int(os.getenv("SERVER_PORT", "8000")),
        reload=True,
        log_level="info",
        timeout_keep_alive=30
    )