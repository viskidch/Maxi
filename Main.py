from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, ForeignKey, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv
import os
import json
from twilio.rest import Client as TwilioClient
from ai_maxi import moderate_message, generate_recommendations  # Import our AI module

# Load environment variables
load_dotenv()

# App Setup
app = FastAPI(title="VictorChat", version="2.0", description="Blue & White Chat App")

# CORS (allow frontend domains only in production!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///victorchat.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ------------------------------
# DATABASE MODELS
# ------------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, index=True)
    phone = Column(String, unique=True, index=True)
    email = Column(EmailStr, unique=True, index=True)
    name = Column(String, index=True)
    hashed_password = Column(String)
    profile_pic = Column(String, default="https://via.placeholder.com/40")
    created_at = Column(DateTime, default=datetime.utcnow)
    is_verified = Column(Boolean, default=False)


class Contact(Base):
    __tablename__ = "contacts"
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"))
    contact_phone = Column(String, index=True)
    contact_name = Column(String)
    is_saved = Column(Boolean, default=False)


class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, index=True)
    sender_id = Column(String, ForeignKey("users.id"))
    receiver_id = Column(String, ForeignKey("users.id"))
    content = Column(String)
    media_url = Column(String, nullable=True)
    is_voice_note = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="sent")  # sent/delivered/read
    is_safe = Column(Boolean, default=True)


class CallLog(Base):
    __tablename__ = "call_logs"
    id = Column(String, primary_key=True, index=True)
    caller_id = Column(String, ForeignKey("users.id"))
    receiver_id = Column(String, ForeignKey("users.id"))
    call_type = Column(String)  # voice/video
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    duration = Column(Float)


class StatusUpdate(Base):
    __tablename__ = "status_updates"
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"))
    content = Column(String)
    media_url = Column(String, nullable=True)
    expires_at = Column(DateTime, default=datetime.utcnow() + timedelta(hours=24))
    views = Column(JSON, default=list)


# Create tables (run once)
Base.metadata.create_all(bind=engine)


# ------------------------------
# AUTHENTICATION UTILS
# ------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# ------------------------------
# DEPENDENCIES
# ------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ------------------------------
# PYDANTIC SCHEMAS
# ------------------------------
class UserCreate(BaseModel):
    phone: str = Field(..., example="+1234567890")
    email: EmailStr
    name: str = Field(..., min_length=2)
    password: str = Field(..., min_length=6)

class MessageCreate(BaseModel):
    receiver_id: str
    content: str
    media_url: str | None = None
    is_voice_note: bool = False


# ------------------------------
# API ROUTES
# ------------------------------
@app.post("/signup", status_code=status.HTTP_201_CREATED)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    db_user = db.query(User).filter((User.phone == user.phone) | (User.email == user.email)).first()
    if db_user:
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Create new user
    hashed_pw = get_password_hash(user.password)
    new_user = User(
        id=os.urandom(16).hex(),
        phone=user.phone,
        email=user.email,
        name=user.name,
        hashed_password=hashed_pw
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Generate token
    access_token = create_access_token(data={"sub": new_user.id})
    return {"access_token": access_token, "token_type": "bearer", "user": new_user}


@app.post("/login")
def login(phone: str, password: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone == phone).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": user.id})
    return {"access_token": access_token, "token_type": "bearer"}


# ------------------------------
# WEBSOCKET FOR REAL-TIME CHAT
# ------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}  # user_id: websocket

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        del self.active_connections[user_id]

    async def broadcast(self, message: dict, sender_id: str, receiver_id: str | None = None):
        """Send message to specific receiver or broadcast"""
        if receiver_id and receiver_id in self.active_connections:
            await self.active_connections[receiver_id].send_json(message)
        else:
            for user_id, connection in self.active_connections.items():
                if user_id != sender_id:
                    await connection.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, db: Session = Depends(get_db)):
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_json()
            # AI Moderation
            moderation = moderate_message(data["content"])
            if not moderation["safe"]:
                await websocket.send_json({"error": moderation["reason"]})
                continue
            
            # Save message to DB
            new_msg = Message(
                id=os.urandom(16).hex(),
                sender_id=user_id,
                receiver_id=data["receiver_id"],
                content=data["content"],
                media_url=data.get("media_url"),
                is_voice_note=data.get("is_voice_note", False),
                is_safe=True
            )
            db.add(new_msg)
            db.commit()
            
            # Broadcast message
            await manager.broadcast({
                "id": new_msg.id,
                "sender": user_id,
                "content": new_msg.content,
                "timestamp": new_msg.timestamp.isoformat(),
                "media_url": new_msg.media_url
            }, sender_id=user_id, receiver_id=data["receiver_id"])
    except WebSocketDisconnect:
        manager.disconnect(user_id)


# ------------------------------
# STATUS ROUTES
# ------------------------------
@app.post("/status")
def create_status(status: str, media_url: str | None = None, db: Session = Depends(get_db), user_id: str = Depends(get_current_user)):
    new_status = StatusUpdate(
        id=os.urandom(16).hex(),
        user_id=user_id,
        content=status,
        media_url=media_url,
        expires_at=datetime.utcnow() + timedelta(hours=24)
    )
    db.add(new_status)
    db.commit()
    return {"status_id": new_status.id, "expires_at": new_status.expires_at}


@app.get("/status/feed")
def get_status_feed(db: Session = Depends(get_db), user_id: str = Depends(get_current_user)):
    # Get statuses from contacts
    contacts = db.query(Contact).filter(Contact.user_id == user_id).all()
    contact_ids = [c.contact_phone for c in contacts]
    statuses = db.query(StatusUpdate).filter(
        StatusUpdate.user_id.in_(contact_ids),
        StatusUpdate.expires_at > datetime.utcnow()
    ).all()
    return statuses


# ------------------------------
# CALL ROUTES (Twilio Integration)
# ------------------------------
@app.post("/call/initiate")
def initiate_call(caller_id: str, receiver_id: str, call_type: str, db: Session = Depends(get_db)):
    # Get user phone numbers
    caller = db.query(User).filter(User.id == caller_id).first()
    receiver = db.query(User).filter(User.id == receiver_id).first()
    
    # Use Twilio for call initiation
    twilio_client = TwilioClient(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
    call = twilio_client.calls.create(
        twiml=f'<Response><Dial>{receiver.phone}</Dial></Response>',
        from_=caller.phone,
        to=receiver.phone
    )
    
    # Log call
    call_log = CallLog(
        id=os.urandom(16).hex(),
        caller_id=caller_id,
        receiver_id=receiver_id,
        call_type=call_type,
        start_time=datetime.utcnow(),
        status="initiated"
    )
    db.add(call_log)
    db.commit()
    return {"call_sid": call.sid, "status": "initiated"}


# ------------------------------
# RUN APP
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
