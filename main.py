from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, HTTPBearer
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
import os
import fitz
import re
from database import SessionLocal, Base, engine
from models import User, Document
from auth import hash_password, verify_password, create_access_token, verify_access_token
import boto3
from botocore.exceptions import ClientError
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from openai import OpenAI
from uuid import uuid4
from transformers import pipeline

# ------------------------------
# Load Local Transformer Pipelines
# ------------------------------
print("Loading local summarization and chat models...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
chat_model = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")

# ------------------------------
# Database Setup
# ------------------------------
Base.metadata.create_all(bind=engine)
oauth2_scheme = HTTPBearer()
app = FastAPI()

# ------------------------------
# CORS Setup
# ------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://case-management-frontend.s3-website-us-east-1.amazonaws.com",
        "http://case-management-frontend.s3-website-us-east-1.amazonaws.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# File + Vector DB Setup
# ------------------------------
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

qdrant = QdrantClient(":memory:")
collection_name = "user_docs"
qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# ------------------------------
# Local Model Helpers
# ------------------------------
def local_summarize(text: str) -> str:
    """Use the local summarization model."""
    try:
        summary = summarizer(text, max_length=300, min_length=50, do_sample=False)
        return summary[0]["summary_text"].strip()
    except Exception as e:
        return f"Local summarization error: {str(e)}"

def local_chat(prompt: str, history=None) -> str:
    """Use the local chat model (simple text generation)."""
    try:
        context = ""
        if history:
            context = "\n".join([f"User: {h.user}\nBot: {h.bot}" for h in history])
        full_prompt = f"{context}\nUser: {prompt}\nAssistant:"
        response = chat_model(full_prompt, max_new_tokens=200, temperature=0.6)
        return response[0]["generated_text"].split("Assistant:")[-1].strip()
    except Exception as e:
        return f"Local chat error: {str(e)}"

# ------------------------------
# OpenAI Client for Embeddings
# ------------------------------
client2 = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------
# Pydantic Schemas
# ------------------------------
class UserCreate(BaseModel):
    email: str
    password: str

class UserOut(BaseModel):
    email: str
    public_data: bool

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatHistoryItem(BaseModel):
    user: str
    bot: str

class ChatRequest(BaseModel):
    user_message: str
    chat_history: list[ChatHistoryItem] = []

class EmailRequest(BaseModel):
    to_email: EmailStr

# ------------------------------
# DB Session
# ------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ------------------------------
# Utility Functions
# ------------------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    doc.close()
    return text

def clean_text(text):
    text = text.replace("Â", " ")
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

def send_email(to_address: str, subject: str, body_text: str):
    ses = boto3.client('ses', region_name='us-east-1')
    try:
        response = ses.send_email(
            Source="ohrivibhav@gmail.com",
            Destination={"ToAddresses": [to_address]},
            Message={
                "Subject": {"Data": subject},
                "Body": {"Text": {"Data": body_text}}
            }
        )
        return response
    except ClientError as e:
        raise Exception(f"Email failed to send: {e.response['Error']['Message']}")

def embed_text(text: str):
    response = client2.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def retrieve_relevant_docs(user_email: str, query: str, db: Session):
    query_embedding = embed_text(query)

    public_users = db.query(User.email).filter(User.public_data == True).all()
    public_emails = [u.email for u in public_users]

    search_filter = Filter(
        should=[
            FieldCondition(key="user_email", match=MatchValue(value=user_email))
        ] + [
            FieldCondition(key="user_email", match=MatchValue(value=email))
            for email in public_emails if email != user_email
        ]
    )

    search_result = qdrant.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=5,
        query_filter=search_filter
    )

    retrieved = []
    for hit in search_result:
        payload = hit.payload or {}
        content_type = payload.get("type", "summary")
        snippet = payload.get("content", "")
        filename = payload.get("filename", "Unknown File")
        if snippet:
            retrieved.append(f"[{filename} - {content_type.upper()}]\n{snippet}\n")
    return retrieved

# ------------------------------
# Middleware
# ------------------------------
@app.middleware("http")
async def log_requests(request, call_next):
    print(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    return response

# ------------------------------
# Auth + User Routes
# ------------------------------
@app.post("/register/")
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = hash_password(user.password)
    new_user = User(email=user.email, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    return {"msg": "User registered successfully"}

@app.post("/login/", response_model=Token)
async def login_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user is None or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(data={"sub": db_user.email})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/me/", response_model=UserOut)
async def read_users_me(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    payload = verify_access_token(token.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.email == payload["sub"]).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {"email": user.email, "public_data": user.public_data}

# ------------------------------
# File Upload + Summarization
# ------------------------------
@app.post("/upload-openai/")
async def upload_pdfs_openai(
    files: list[UploadFile] = File(...),
    user_prompt: str = Form(""),
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    payload = verify_access_token(token.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    user_email = payload["sub"]

    all_text = ""
    file_outputs = []

    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        extracted_text = extract_text_from_pdf(file_path)
        cleaned_text = clean_text(extracted_text)
        all_text += f"--- Start of {file.filename} ---\n{cleaned_text}\n\n"

        # Local summarization instead of Groq/OpenAI
        summary_output = local_summarize(cleaned_text)

        doc = Document(filename=file.filename, user_email=user_email, summary=summary_output)
        db.add(doc)
        db.commit()
        db.refresh(doc)

        try:
            summary_embedding = embed_text(summary_output)
            raw_embedding = embed_text(cleaned_text)
            qdrant.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=str(uuid4()),
                        vector=summary_embedding,
                        payload={
                            "user_email": user_email,
                            "filename": file.filename,
                            "type": "summary",
                            "content": summary_output
                        }
                    ),
                    PointStruct(
                        id=str(uuid4()),
                        vector=raw_embedding,
                        payload={
                            "user_email": user_email,
                            "filename": file.filename,
                            "type": "raw_text",
                            "content": cleaned_text
                        }
                    )
                ]
            )
        except Exception as e:
            print(f"Embedding error for {file.filename}: {e}")

        file_outputs.append({
            "filename": file.filename,
            "extraction_and_summary": summary_output
        })

    # Consolidated summary using local summarizer
    consolidated_summary = local_summarize(all_text)

    return {
        "per_file_outputs": file_outputs,
        "consolidated_summary": consolidated_summary
    }

# ------------------------------
# Chat Endpoint (local model)
# ------------------------------
@app.post("/chat/")
async def chat_with_bot(
    chat_request: ChatRequest,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    payload = verify_access_token(token.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_email = payload["sub"]
    relevant_docs = retrieve_relevant_docs(user_email, chat_request.user_message, db)
    context = "\n\n".join(relevant_docs)
    prompt = f"Context:\n{context}\n\nUser: {chat_request.user_message}"

    bot_reply = local_chat(prompt, chat_request.chat_history)
    return {"response": bot_reply}

# ------------------------------
# Email Sending with PDF Attachment
# ------------------------------
@app.post("/send-report-email/")
async def send_report_email(
    to_email: str = Form(...),
    report_pdf: UploadFile = File(...),
    token: str = Depends(oauth2_scheme)
):
    payload = verify_access_token(token.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    try:
        ses = boto3.client('ses', region_name='us-east-1')
        file_bytes = await report_pdf.read()
        attachment = {'Filename': report_pdf.filename, 'Data': file_bytes}

        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.application import MIMEApplication

        msg = MIMEMultipart()
        msg['Subject'] = 'Your Error Ticket Summary Report'
        msg['From'] = 'ohrivibhav@gmail.com'
        msg['To'] = to_email

        body = MIMEText('Attached is the summary report you generated.', 'plain')
        msg.attach(body)

        part = MIMEApplication(attachment["Data"])
        part.add_header('Content-Disposition', 'attachment', filename=attachment["Filename"])
        msg.attach(part)

        ses.send_raw_email(
            Source=msg['From'],
            Destinations=[msg['To']],
            RawMessage={'Data': msg.as_string()}
        )
        return {"message": "Email with report sent successfully."}
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"Email failed: {e.response['Error']['Message']}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# ------------------------------
# Toggle Public Data
# ------------------------------
@app.post("/toggle-public/")
def toggle_public_data(public: bool, token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    payload = verify_access_token(token.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.email == payload["sub"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.public_data = public
    db.commit()
    return {"message": f"Public data access set to {public}"}
