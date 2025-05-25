from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, HTTPBearer
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
import os
import fitz
import re
from openai import OpenAI
from database import SessionLocal, Base, engine
from models import User
from auth import hash_password, verify_password, create_access_token, verify_access_token
import boto3
from botocore.exceptions import ClientError
from collections import defaultdict

user_uploaded_text = defaultdict(str)

client = OpenAI(
    api_key="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"
)

Base.metadata.create_all(bind=engine)

oauth2_scheme = HTTPBearer()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://case-management-frontend.s3-website-us-east-1.amazonaws.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class UserCreate(BaseModel):
    email: str
    password: str

class UserOut(BaseModel):
    email: str

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatRequest(BaseModel):
    user_message: str

class EmailRequest(BaseModel):
    to_email: EmailStr

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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

@app.post("/register/")
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = hash_password(user.password)
    new_user = User(email=user.email, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"msg": "User registered successfully"}

@app.post("/login/", response_model=Token)
async def login_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user is None or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": db_user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/me/", response_model=UserOut)
async def read_users_me(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    payload = verify_access_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.email == payload["sub"]).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/upload-openai/")
async def upload_pdfs_openai(files: list[UploadFile] = File(...), token: str = Depends(oauth2_scheme)):
    payload = verify_access_token(token)
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
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. Please extract the following fields: "
                            "incident number, system, date, reporter, priority, responsible area, and solution "
                            "from the text below. Then provide a brief summary of the text.\n\n" + cleaned_text
                        )
                    }
                ],
                max_tokens=500,
                temperature=0.5
            )
            output = response.choices[0].message.content.strip()
        except Exception as e:
            output = f"OpenAI API error while processing {file.filename}: {str(e)}"
        file_outputs.append({
            "filename": file.filename,
            "extraction_and_summary": output
        })
    user_uploaded_text[user_email] = all_text
    try:
        consolidated_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Summarize the following consolidated text from multiple documents:\n\n" + all_text
                    )
                }
            ],
            max_tokens=300,
            temperature=0.5
        )
        consolidated_summary = consolidated_response.choices[0].message.content.strip()
    except Exception as e:
        consolidated_summary = f"Error summarizing consolidated text: {str(e)}"
    return {
        "per_file_outputs": file_outputs,
        "consolidated_summary": consolidated_summary
    }

@app.post("/chat/")
async def chat_with_bot(chat_request: ChatRequest, token: str = Depends(oauth2_scheme)):
    payload = verify_access_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    user_email = payload["sub"]
    context = user_uploaded_text.get(user_email, "")
    truncated_context = context[:4000]
    prompt_messages = [
        {"role": "system", "content": "You are a helpful legal assistant. Use the uploaded document context to answer questions accurately."},
        {"role": "user", "content": f"Document context:\n{truncated_context}"},
        {"role": "user", "content": chat_request.user_message}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=prompt_messages,
            temperature=0.3,
            max_tokens=500
        )
        bot_reply = response.choices[0].message.content.strip()
        return {"response": bot_reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/send-email/")
def email_user(request: EmailRequest):
    send_email(
        to_address=request.to_email,
        subject="Welcome to Our App",
        body_text="Thanks for signing up!"
    )
    return {"message": "Email sent"}
