from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os, json, uuid
from gtts import gTTS
import asyncio
import aiohttp
from typing import Optional, List, Dict
from dotenv import load_dotenv
import google.generativeai as genai

from sentence_transformers import SentenceTransformer, util
import numpy as np

load_dotenv()

# ---------------- FastAPI setup ----------------
app = FastAPI(title="🎙️ Aygen Voice AI Assistant Pro")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Gemini Configuration ----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        print("✅ gemini-2.0-flash Latest loaded")
    except Exception as e:
        try:
            gemini_model = genai.GenerativeModel('gemini-pro')
            print("✅ Gemini Pro loaded")
        except Exception as e2:
            gemini_model = None
            print(f"❌ Gemini failed: {e2}")
else:
    gemini_model = None
    print("⚠️ No Gemini API key found in .env file")

# ---------------- Load Portfolio Dataset ----------------
DATA_PATH = "portfolio_qa.json"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("❌ portfolio_qa.json not found")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

questions = [d['q'] for d in data]
answers = [d['a'] for d in data]

# ---------------- Load Sentence Transformer ----------------
print("🧠 Loading Sentence Transformer...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
question_embeddings = model.encode(questions, convert_to_tensor=True)

# ---------------- Voice setup ----------------
VOICE_DIR = "voices"
os.makedirs(VOICE_DIR, exist_ok=True)

# ---------------- User Memory Storage ----------------
USER_SESSIONS = {}  # In-memory user session storage

# ---------------- Schema ----------------
class UserData(BaseModel):
    name: str
    email: str
    company: Optional[str] = None
    purpose: Optional[str] = None
    visitCount: int = 1
    firstVisit: Optional[str] = None
    lastVisit: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    message: str

class ChatRequest(BaseModel):
    user_message: str
    chat_history: List[ChatMessage]
    user_data: Optional[UserData] = None

# ---------------- Helper Functions ----------------

def is_portfolio_question(query: str) -> bool:
    """Check if question is about Yogesh's portfolio"""
    portfolio_keywords = [
        "yogesh", "baghel", "portfolio", "project", "skill", "experience", 
        "resume", "education", "contact", "techbooklibrary", "arimart", 
        "sapeagleerp", "sofnotek", "nucleous", "work", "job", "developer",
        "email", "phone", "linkedin", "github", "hire", "cv"
    ]
    return any(keyword in query.lower() for keyword in portfolio_keywords)

def find_portfolio_answer(query: str, threshold: float = 0.5) -> Optional[str]:
    """Find best matching answer from portfolio using semantic search"""
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, question_embeddings)[0]
    
    best_idx = np.argmax(cos_scores.cpu().numpy())
    best_score = cos_scores[best_idx].item()
    
    if best_score > threshold:
        print(f"✅ Portfolio match: {questions[best_idx]} (score: {best_score:.2f})")
        return answers[best_idx]
    
    return None

async def generate_gemini_response(query: str, context: str, chat_history: list, user_data: Optional[UserData] = None) -> str:
    """Generate personalized response using Gemini with user context"""
    if not gemini_model:
        return "Gemini API not configured. Please add GEMINI_API_KEY to your .env file."
    
    try:
        # Build user context
        user_context = ""
        if user_data:
            user_context = f"""
USER PROFILE:
- Name: {user_data.name}
- Email: {user_data.email}
- Company: {user_data.company or 'Not specified'}
- Purpose: {user_data.purpose or 'Not specified'}
- Visit Count: {user_data.visitCount}
- Status: {'Returning visitor' if user_data.visitCount > 1 else 'First-time visitor'}
"""
        
        # Build conversation history - handle both dict and Pydantic objects
        history_text = ""
        for msg in chat_history[-6:]:  # Last 6 messages for better context
            # Handle both ChatMessage objects and dicts
            if isinstance(msg, dict):
                role = msg.get('role', 'user')
                message = msg.get('message', '')
            else:
                # Pydantic object
                role = msg.role
                message = msg.message
            history_text += f"{role.upper()}: {message}\n"
        
        # Create personalized prompt
        prompt = f"""You are Aygen, a friendly and professional AI assistant for Yogesh Baghel's portfolio.

{user_context}

PORTFOLIO INFORMATION:
{context}

CONVERSATION HISTORY:
{history_text}

USER QUERY: {query}

INSTRUCTIONS:
1. **Personalization**: Address the user by name ({user_data.name if user_data else 'there'})
2. If this is a returning visitor (visit #{user_data.visitCount if user_data else 1}), acknowledge it warmly
3. If user mentioned their company or purpose, reference it naturally when relevant
4. For portfolio questions, use the PORTFOLIO INFORMATION above
5. For general questions, answer based on your knowledge
6. Be conversational, professional, and helpful
7. Keep responses concise (under 150 words) unless more detail is requested
8. If discussing Yogesh's work, relate it to the user's purpose/interest when possible

Answer:"""

        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return "I'm having trouble generating a response. Please try again."

def generate_voice(text: str):
    """Generate voice from text"""
    lang = "hi" if any(ch in text for ch in "कखगघचछजझटठडढतथदधनपफबभमयरलवशषसह") else "en"
    tts = gTTS(text=text, lang=lang)
    file_name = f"{uuid.uuid4()}.mp3"
    path = os.path.join(VOICE_DIR, file_name)
    tts.save(path)
    return path

def store_user_session(user_data: UserData, chat_history: list):
    """Store user session in memory"""
    USER_SESSIONS[user_data.email] = {
        'user_data': user_data.dict(),
        'chat_history': chat_history,
        'last_active': user_data.lastVisit
    }

# ---------------- Main Chat Route with Memory ----------------
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        user_msg = request.user_message.strip()
        user_data = request.user_data
        
        # Log user info
        if user_data:
            print(f"\n💬 {user_data.name} ({user_data.email}): {user_msg}")
            print(f"📊 Visit #{user_data.visitCount}")
        else:
            print(f"\n💬 Anonymous: {user_msg}")
        
        # Extract actual query (remove user context prefix if present)
        actual_query = user_msg
        if user_msg.startswith('[User:'):
            actual_query = user_msg.split(']', 1)[1].strip() if ']' in user_msg else user_msg
        
        # Check if it's a portfolio question
        is_portfolio = is_portfolio_question(actual_query)
        
        # Try to find direct portfolio answer
        portfolio_answer = find_portfolio_answer(actual_query) if is_portfolio else None
        
        # Determine answer strategy
        if portfolio_answer and not user_data:
            # Direct portfolio match, no personalization needed
            answer = portfolio_answer
        
        elif gemini_model:
            # Use Gemini for personalized response
            if is_portfolio:
                print("🤖 Using Gemini with portfolio context + personalization")
                portfolio_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in zip(questions[:10], answers[:10])])
            else:
                print("🤖 Using Gemini for general query + personalization")
                portfolio_context = ""
            
            answer = await generate_gemini_response(
                actual_query, 
                portfolio_context, 
                request.chat_history,
                user_data
            )
        
        else:
            # Fallback
            if portfolio_answer:
                answer = portfolio_answer
            else:
                answer = "I can answer questions about Yogesh Baghel's portfolio. Please configure GEMINI_API_KEY for personalized responses."
        
        # Store session if user data provided
        if user_data:
            store_user_session(user_data, request.chat_history + [
                {'role': 'user', 'message': user_msg},
                {'role': 'assistant', 'message': answer}
            ])
        
        # Generate voice
        audio_path = generate_voice(answer)
        audio_url = f"/voice/{os.path.basename(audio_path)}"
        
        print(f"✅ Response: {answer[:100]}...")
        
        return {"answer": answer, "audio_url": audio_url}
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/voice/{file_name}")
async def get_voice(file_name: str):
    path = os.path.join(VOICE_DIR, file_name)
    if os.path.exists(path):
        return FileResponse(path, media_type="audio/mpeg")
    return {"error": "Voice file not found"}

@app.get("/user/{email}/history")
async def get_user_history(email: str):
    """Retrieve user's chat history"""
    session = USER_SESSIONS.get(email)
    if session:
        return {
            "found": True,
            "user_data": session['user_data'],
            "chat_history": session['chat_history'],
            "last_active": session['last_active']
        }
    return {"found": False}

@app.get("/stats")
async def get_stats():
    """Get usage statistics"""
    return {
        "total_users": len(USER_SESSIONS),
        "active_sessions": sum(1 for s in USER_SESSIONS.values() if s.get('chat_history')),
        "gemini_enabled": gemini_model is not None
    }

@app.get("/config")
async def get_config():
    return {
        "gemini_enabled": gemini_model is not None,
        "user_sessions": len(USER_SESSIONS)
    }

@app.get("/")
async def root():
    return {
        "message": "🎙️ Aygen Voice AI Assistant Pro with Memory",
        "gemini_status": "✅ Enabled" if gemini_model else "❌ Disabled",
        "features": [
            "User authentication & memory",
            "Personalized responses",
            "Chat history storage",
            "Voice synthesis"
        ],
        "endpoints": {
            "/chat": "POST - Main chat with user context",
            "/user/{email}/history": "GET - Get user's chat history",
            "/stats": "GET - Usage statistics",
            "/config": "GET - Check configuration"
        }
    }

if __name__ == "__main__":
    print("🚀 Aygen Pro Assistant with Memory Starting...")
    print(f"🤖 Gemini: {'✅ Enabled' if gemini_model else '❌ Not configured'}")
    print(f"💾 User Memory: ✅ Enabled")
    
    if not gemini_model:
        print("\n⚠️  SETUP REQUIRED:")
        print("1. Create a file named '.env' in your project folder")
        print("2. Add this line: GEMINI_API_KEY=your-api-key-here")
        print("3. Get free API key: https://aistudio.google.com/apikey\n")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)