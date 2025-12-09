from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
import os
from io import BytesIO
from pathlib import Path
import uuid
import tarfile
import shutil
import threading
import time
from datetime import datetime
from enum import Enum

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not found. Please install it using:")
    print("  pip install pymupdf")
    sys.exit(1)

try:
    from PIL import Image
    import pytesseract
except ImportError:
    print("OCR dependencies not found. Please install them using:")
    print("  pip install pillow pytesseract")
    sys.exit(1)

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    print("ChromaDB not found. Please install it using:")
    print("  pip install chromadb")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("OpenAI not found. Please install it using:")
    print("  pip install openai")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv not found. Please install it using:")
    print("  pip install python-dotenv")
    sys.exit(1)

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("boto3 not found. Please install it using:")
    print("  pip install boto3")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("requests not found. Please install it using:")
    print("  pip install requests")
    sys.exit(1)

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
except ImportError:
    print("reportlab not found. Please install it using:")
    print("  pip install reportlab")
    sys.exit(1)

try:
    from twilio.rest import Client as TwilioClient
except ImportError:
    print("twilio not found. Please install it using:")
    print("  pip install twilio")
    sys.exit(1)

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    from bson import ObjectId
    MONGODB_AVAILABLE = True
except ImportError:
    print("pymongo not found. Please install it using:")
    print("  pip install pymongo")
    MONGODB_AVAILABLE = False

try:
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders
except ImportError:
    print("Email libraries not found. Please ensure Python's email libraries are available.")
    sys.exit(1)

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    import pickle
    GOOGLE_CALENDAR_AVAILABLE = True
except ImportError:
    print("Google Calendar libraries not found. Please install them using:")
    print("  pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")
    GOOGLE_CALENDAR_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

# Configure Tesseract path for Windows
TESSERACT_FOUND = False
if sys.platform == 'win32':
    import shutil
    tesseract_path = shutil.which('tesseract')
    if tesseract_path:
        TESSERACT_FOUND = True
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    else:
        common_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME')),
        ]
        for path in common_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                TESSERACT_FOUND = True
                break
else:
    TESSERACT_FOUND = True

# Initialize FastAPI
app = FastAPI(
    title="PDF Text Extraction & Query API with WhatsApp",
    description="Extract text from PDFs, store in ChromaDB, query with GPT-4o, and send via WhatsApp",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job tracking for async extraction
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# In-memory job storage (could be replaced with Redis/MongoDB for production)
extraction_jobs = {}
job_lock = threading.Lock()
backup_lock = threading.Lock()  # Lock to prevent concurrent ChromaDB backups
extraction_in_progress = False  # Flag to track if extraction is in progress
extraction_lock = threading.Lock()  # Lock to protect extraction_in_progress flag


@app.head("/health")
async def health_check():
    """
    Health check endpoint (HEAD request).
    Returns 200 OK if the service is running.
    """
    return None


# Google Calendar configuration
CALENDAR_SCOPES = ['https://www.googleapis.com/auth/calendar']

# Initialize OpenAI client (with error handling)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("\n⚠️  WARNING: OPENAI_API_KEY not found in .env file or environment!")
    print("The /query endpoint and embeddings will not work without an API key.")
    openai_client = None
    openai_ef = None
else:
    openai_client = OpenAI(api_key=openai_api_key)
    # Create OpenAI embedding function for ChromaDB
    try:
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-ada-002"  # Using ada-002 as default, can be changed to text-embedding-3-small or text-embedding-3-large
        )
    except Exception as e:
        print(f"\n⚠️  WARNING: Failed to initialize OpenAI embeddings: {e}")
        openai_ef = None

# Initialize S3 client (with error handling) - MUST be before ChromaDB restore
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "velswidget")

def get_bucket_region(bucket_name, access_key, secret_key, default_region):
    """Get the actual region of an S3 bucket."""
    try:
        # Create a temporary client with default region to get bucket location
        temp_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=default_region
        )
        # Get bucket location (returns 'us-east-1' as None or the actual region)
        response = temp_client.get_bucket_location(Bucket=bucket_name)
        location = response.get('LocationConstraint')
        # If location is None or empty, it means us-east-1
        if location is None or location == '':
            return 'ap-south-1'
        return location
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        # If bucket doesn't exist or access denied, return default
        if error_code in ['NoSuchBucket', 'AccessDenied', '403']:
            print(f"  ⚠ Could not detect bucket region ({error_code}), using configured: {default_region}")
            return default_region
        # For other errors, try common regions
        print(f"  ⚠ Error detecting bucket region: {e}")
        return default_region
    except Exception as e:
        print(f"  ⚠ Could not detect bucket region: {e}, using configured: {default_region}")
        return default_region

if not aws_access_key or not aws_secret_key:
    print("\n⚠️  WARNING: AWS credentials not found in .env file or environment!")
    print("The S3 upload functionality will not work without AWS credentials.")
    s3_client = None
else:
    # Detect the actual bucket region to avoid signature mismatches
    print(f"\n🔍 Detecting S3 bucket region for '{S3_BUCKET_NAME}'...")
    detected_region = get_bucket_region(S3_BUCKET_NAME, aws_access_key, aws_secret_key, aws_region)
    if detected_region != aws_region:
        print(f"  ✓ Bucket region detected: {detected_region} (was configured as: {aws_region})")
        aws_region = detected_region
    else:
        print(f"  ✓ Using configured region: {aws_region}")
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )

CHROMADB_S3_KEY = "chromadb_backup/chromadb.tar.gz"  # S3 key for ChromaDB backup
CHROMADB_LOCAL_PATH = "./chroma_db"

def download_chromadb_from_s3():
    """Download ChromaDB backup from S3 if it exists."""
    if s3_client is None:
        print("\n⚠️  S3 client not configured, skipping ChromaDB restore from S3")
        return False
    
    try:
        print("\n📥 Checking for ChromaDB backup in S3...")
        # Check if backup exists
        try:
            s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=CHROMADB_S3_KEY)
            print(f"  ✓ Found ChromaDB backup in S3")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                print(f"  ⚠ No ChromaDB backup found in S3 (this is normal for first run)")
                return False
            else:
                raise
        
        # Download the backup
        print(f"  📥 Downloading ChromaDB backup from S3...")
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=CHROMADB_S3_KEY)
        backup_data = response['Body'].read()
        
        # Remove existing chroma_db directory if it exists
        if os.path.exists(CHROMADB_LOCAL_PATH):
            print(f"  🗑️  Removing existing local ChromaDB directory...")
            shutil.rmtree(CHROMADB_LOCAL_PATH)
        
        # Extract the tar.gz file
        print(f"  📦 Extracting ChromaDB backup...")
        with tarfile.open(fileobj=BytesIO(backup_data), mode='r:gz') as tar:
            tar.extractall(path='.')
        
        print(f"  ✓ ChromaDB restored from S3 successfully")
        return True
        
    except Exception as e:
        print(f"  ⚠ Failed to restore ChromaDB from S3: {str(e)}")
        print(f"  → Continuing with empty ChromaDB (this is normal for first run)")
        return False


def upload_chromadb_to_s3(skip_if_extraction_in_progress=False):
    """
    Upload ChromaDB directory to S3 as a backup.
    
    Args:
        skip_if_extraction_in_progress: If True, skip backup if extraction is in progress
    """
    if s3_client is None:
        return False
    
    if not os.path.exists(CHROMADB_LOCAL_PATH):
        return False
    
    # Check if extraction is in progress and skip if requested
    if skip_if_extraction_in_progress:
        with extraction_lock:
            if extraction_in_progress:
                print(f"  ⏭️  Skipping ChromaDB backup (extraction in progress)")
                return False
    
    # Acquire backup lock to prevent concurrent backups
    if not backup_lock.acquire(blocking=False):
        print(f"  ⏭️  Skipping ChromaDB backup (another backup in progress)")
        return False
    
    try:
        print(f"\n📤 Backing up ChromaDB to S3...")
        
        # Create a temporary tar.gz file in memory
        tar_buffer = BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
            tar.add(CHROMADB_LOCAL_PATH, arcname=os.path.basename(CHROMADB_LOCAL_PATH))
        
        tar_buffer.seek(0)
        
        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=CHROMADB_S3_KEY,
            Body=tar_buffer.getvalue(),
            ContentType='application/gzip'
        )
        
        print(f"  ✓ ChromaDB backed up to S3 successfully")
        return True
        
    except Exception as e:
        print(f"  ⚠ Failed to backup ChromaDB to S3: {str(e)}")
        return False
    finally:
        backup_lock.release()


# Download ChromaDB from S3 on startup (if available)
download_chromadb_from_s3()

# Ensure chroma_db directory exists (create if it doesn't)
os.makedirs(CHROMADB_LOCAL_PATH, exist_ok=True)

# Initialize ChromaDB with OpenAI embeddings
chroma_client = chromadb.PersistentClient(path=CHROMADB_LOCAL_PATH)

# Get or create collection with OpenAI embeddings (if available)
if openai_ef:
    collection = chroma_client.get_or_create_collection(
        name="pdf_documents",
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )
    print("\n✓ ChromaDB initialized with OpenAI embeddings (text-embedding-ada-002)")
else:
    # Fallback to default embeddings if OpenAI is not configured
    collection = chroma_client.get_or_create_collection(
        name="pdf_documents",
        metadata={"hnsw:space": "cosine"}
    )
    print("\n⚠️  ChromaDB initialized with default embeddings (OpenAI not configured)")

# Perform initial backup after initialization
if s3_client is not None:
    print("\n📤 Performing initial ChromaDB backup to S3...")
    threading.Thread(target=upload_chromadb_to_s3, daemon=True).start()


def periodic_chromadb_backup():
    """Periodically backup ChromaDB to S3 every 5 minutes. Skips if extraction is in progress."""
    while True:
        time.sleep(300)  # 5 minutes
        if s3_client is not None:
            # Skip backup if extraction is in progress to avoid conflicts
            upload_chromadb_to_s3(skip_if_extraction_in_progress=True)


# Start periodic backup thread
if s3_client is not None:
    backup_thread = threading.Thread(target=periodic_chromadb_backup, daemon=True)
    backup_thread.start()
    print("✓ Periodic ChromaDB backup thread started (every 5 minutes)")

# Initialize Twilio client (with error handling)
twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")

if not twilio_account_sid or not twilio_auth_token:
    print("\n⚠️  WARNING: Twilio credentials not found in .env file or environment!")
    print("The WhatsApp sending functionality will not work without Twilio credentials.")
    twilio_client = None
else:
    twilio_client = TwilioClient(twilio_account_sid, twilio_auth_token)

# Initialize MongoDB client (with error handling)
mongodb_uri = os.getenv("MONGODB_URI", "")
if not mongodb_uri:
    print("\n⚠️  WARNING: MONGODB_URI not found in .env file or environment!")
    print("The brochure tracking functionality will not work without MongoDB URI.")
    mongodb_client = None
elif not MONGODB_AVAILABLE:
    print("\n⚠️  WARNING: pymongo not installed!")
    print("The brochure tracking functionality will not work without pymongo.")
    mongodb_client = None
else:
    try:
        mongodb_client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        # Test the connection
        mongodb_client.admin.command('ping')
        print("\n✓ MongoDB connection established")
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"\n⚠️  WARNING: Failed to connect to MongoDB: {e}")
        print("The brochure tracking functionality will not work.")
        mongodb_client = None
    except Exception as e:
        print(f"\n⚠️  WARNING: Error connecting to MongoDB: {e}")
        mongodb_client = None

# Pydantic model for query request
class QueryRequest(BaseModel):
    query: str
    number: Optional[str] = None
    email: Optional[str] = None

# Pydantic model for calendar request
class CalendarRequest(BaseModel):
    title: str
    date: str  # Format: YYYY-MM-DD
    start_time: str  # Format: HH:MM
    end_time: str  # Format: HH:MM
    description: str = ""
    location: str = ""


def send_whatsapp_message(to_number: str, summary: str, pdf_url: str = None):
    """
    Send WhatsApp message with summary and optional PDF attachment.
    
    Args:
        to_number: Recipient's phone number (e.g., "918438232949")
        summary: Text summary to send
        pdf_url: Optional URL of PDF to attach
        
    Returns:
        dict: Status of message sending
    """
    if twilio_client is None:
        print(f"  ⚠ Twilio client not configured, skipping WhatsApp message")
        return {"status": "skipped", "reason": "Twilio not configured"}
    
    try:
        print(f"\n  📱 Sending WhatsApp message...")
        print(f"    → To: {to_number}")
        print(f"    → Summary length: {len(summary)} chars")
        print(f"    → PDF URL: {pdf_url if pdf_url else 'None'}")
        print(f"    → PDF attached: {'Yes' if pdf_url else 'No'}")
        
        # Format phone number for WhatsApp
        formatted_number = f"whatsapp:{to_number}" if not to_number.startswith('whatsapp:') else to_number
        
        # Prepare message body - just the summary, no prefix
        message_body = summary
        
        # Create message parameters
        message_params = {
            'from_': twilio_whatsapp_number,
            'body': message_body,
            'to': formatted_number
        }
        
        # Add PDF if available - IMPORTANT: Twilio needs publicly accessible URLs
        if pdf_url:
            # Test if URL is accessible
            print(f"    → Testing PDF URL accessibility...")
            try:
                test_response = requests.head(pdf_url, timeout=5)
                print(f"    → PDF URL status code: {test_response.status_code}")
                if test_response.status_code == 200:
                    message_params['media_url'] = [pdf_url]
                    print(f"    → PDF URL is accessible, adding to message")
                else:
                    print(f"    ⚠ PDF URL returned {test_response.status_code}, may not be accessible to Twilio")
                    message_params['media_url'] = [pdf_url]  # Try anyway
            except Exception as url_test_error:
                print(f"    ⚠ Could not test URL accessibility: {url_test_error}")
                message_params['media_url'] = [pdf_url]  # Try anyway
        
        # Send message
        print(f"    → Sending message via Twilio...")
        message = twilio_client.messages.create(**message_params)
        
        print(f"  ✓ WhatsApp message sent successfully!")
        print(f"    → Message SID: {message.sid}")
        print(f"    → Status: {message.status}")
        print(f"    → Direction: {message.direction}")
        
        # Check for any errors
        if hasattr(message, 'error_code') and message.error_code:
            print(f"    ⚠ Error code: {message.error_code}")
            print(f"    ⚠ Error message: {message.error_message}")
        
        return {
            "status": "success",
            "message_sid": message.sid,
            "twilio_status": message.status,
            "pdf_url_sent": pdf_url
        }
        
    except Exception as e:
        print(f"  ✗ Failed to send WhatsApp message: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }


def send_email_message(to_email: str, summary: str, pdf_url: str = None, query: str = None):
    """
    Send email with summary and optional PDF attachment.
    
    Args:
        to_email: Recipient's email address
        summary: Text summary to send
        pdf_url: Optional URL of PDF to attach
        query: The original query (for subject line)
        
    Returns:
        dict: Status of email sending
    """
    # Get SMTP configuration from environment
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_from_email = os.getenv("SMTP_FROM_EMAIL", smtp_username)
    
    if not smtp_username or not smtp_password:
        print(f"  ⚠ SMTP credentials not configured, skipping email")
        return {"status": "skipped", "reason": "SMTP not configured"}
    
    try:
        print(f"\n  📧 Sending email...")
        print(f"    → To: {to_email}")
        print(f"    → Summary length: {len(summary)} chars")
        print(f"    → PDF URL: {pdf_url if pdf_url else 'None'}")
        print(f"    → PDF attached: {'Yes' if pdf_url else 'No'}")
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = smtp_from_email
        msg['To'] = to_email
        msg['Subject'] = f"Query Response: {query[:50]}" if query else "Query Response"
        
        # Add body to email
        body = f"""
Hello,

Thank you for your query. Please find the detailed information below:

{summary}

"""
        
        if pdf_url:
            body += f"\nA detailed PDF document has been attached for your reference.\n"
            body += f"\nYou can also access it directly here: {pdf_url}\n"
        
        body += """
Best regards,
Vel's University
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach PDF if URL is provided
        if pdf_url:
            try:
                print(f"    → Downloading PDF from URL...")
                pdf_response = requests.get(pdf_url, timeout=30)
                pdf_response.raise_for_status()
                
                # Create attachment
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(pdf_response.content)
                encoders.encode_base64(part)
                
                # Extract filename from URL or use default
                filename = pdf_url.split('/')[-1] if '/' in pdf_url else "query_response.pdf"
                if not filename.endswith('.pdf'):
                    filename = "query_response.pdf"
                
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {filename}'
                )
                msg.attach(part)
                print(f"    → PDF attached: {filename}")
            except Exception as pdf_error:
                print(f"    ⚠ Could not attach PDF: {str(pdf_error)}")
                # Continue without PDF attachment
        
        # Send email
        print(f"    → Connecting to SMTP server...")
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Enable encryption
        server.login(smtp_username, smtp_password)
        
        print(f"    → Sending email...")
        text = msg.as_string()
        server.sendmail(smtp_from_email, to_email, text)
        server.quit()
        
        print(f"  ✓ Email sent successfully!")
        print(f"    → To: {to_email}")
        print(f"    → Subject: {msg['Subject']}")
        
        return {
            "status": "success",
            "to": to_email,
            "subject": msg['Subject'],
            "pdf_attached": pdf_url is not None
        }
        
    except Exception as e:
        print(f"  ✗ Failed to send email: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }


def send_webhook_notification(webhook_url: str, job_id: str, job_data: dict, max_retries: int = 3):
    """
    Send webhook notification with retry logic.
    
    Args:
        webhook_url: URL to send the webhook to
        job_id: Job identifier
        job_data: Data to send in the webhook
        max_retries: Maximum number of retry attempts
    """
    for attempt in range(max_retries):
        try:
            print(f"  📡 Sending webhook notification (attempt {attempt + 1}/{max_retries})...")
            response = requests.post(
                webhook_url,
                json=job_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            print(f"  ✓ Webhook notification sent successfully (status: {response.status_code})")
            return True
        except requests.exceptions.RequestException as e:
            print(f"  ⚠ Webhook notification failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                # Exponential backoff: wait 2^attempt seconds
                wait_time = 2 ** attempt
                print(f"  ⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  ✗ Webhook notification failed after {max_retries} attempts")
                return False
    return False


def send_messages_background(user_number: Optional[str], user_email: Optional[str], summary: str, compiled_pdf_url: Optional[str], query: str):
    """
    Background function to send WhatsApp and Email messages, and track in MongoDB.
    This runs in a separate thread to avoid blocking the API response.
    """
    try:
        print(f"\n  🔄 Background: Starting message sending...")
        
        # Send WhatsApp message (only if number is provided)
        whatsapp_status = {"status": "skipped", "reason": "No phone number provided"}
        if user_number:
            whatsapp_status = send_whatsapp_message(user_number, summary, compiled_pdf_url)
        else:
            print(f"  ⚠ Background: Skipping WhatsApp message (no phone number provided)")
        
        # Send Email message (only if email is provided)
        email_status = {"status": "skipped", "reason": "No email provided"}
        if user_email:
            email_status = send_email_message(user_email, summary, compiled_pdf_url, query)
        else:
            print(f"  ⚠ Background: Skipping email message (no email provided)")
        
        # Track brochure sharing in MongoDB (only if PDF was created and sent successfully)
        # Only track if we have at least one contact method and PDF was created
        if compiled_pdf_url and (user_number or user_email) and (whatsapp_status.get("status") == "success" or email_status.get("status") == "success"):
            print(f"  📊 Background: Tracking brochure sharing...")
            track_brochure_shared(user_number or "", user_email or "", compiled_pdf_url, query)
        
        print(f"  ✅ Background: Message sending completed")
        print(f"    📱 WhatsApp: {whatsapp_status['status']}")
        print(f"    📧 Email: {email_status['status']}")
        
    except Exception as e:
        print(f"  ❌ Background: Error sending messages: {str(e)}")


def normalize_phone_number(phone_number: str) -> str:
    """Normalize phone number to DB format: '91' + 10 digits"""
    # Remove  non-digit characters
    digits_only = ''.join(filter(str.isdigit, phone_number))
    
    # If the number starts with "91" and has more than 10 digits, remove the "91" prefix
    if digits_only.startswith("91") and len(digits_only) > 10:
        phone_number_clean = digits_only[2:]  # Remove first 2 digits (91)
    elif len(digits_only) > 10:
        # If it's longer than 10 digits but doesn't start with 91, take last 10 digits
        phone_number_clean = digits_only[-10:]
    elif len(digits_only) < 10:
        # If less than 10 digits, pad with zeros
        phone_number_clean = digits_only.zfill(10)
    else:
        # Exactly 10 digits
        phone_number_clean = digits_only
    
    # Convert to DB format: "91" + 10 digits
    return f"91{phone_number_clean}" if len(phone_number_clean) == 10 else phone_number


def determine_course_from_query(query: str) -> str:
    """Determine if query is course-specific or general using OpenAI"""
    if not openai_client:
        print("  ⚠ OpenAI client not available, defaulting to 'general'")
        return "general"
    
    try:
        prompt = f"""Analyze the following query and determine if it's related to a SPECIFIC engineering course or is a GENERAL query about the college.

Available courses:
- Computer Science and Engineering
- Information Technology
- Electrical and Electronics Engineering
- Electronics and Communication Engineering
- Mechanical Engineering

If the query is about a SPECIFIC course, return ONLY the exact course name from the list above.
If the query is GENERAL (about college, admissions, fees, facilities, etc. without mentioning a specific course), return "general".

Query: "{query}"

Return ONLY the course name or "general" (no additional text):"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a course classification assistant. Analyze queries and determine if they're course-specific or general."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        course = response.choices[0].message.content.strip()
        
        # Validate course name
        valid_courses = [
            "Computer Science and Engineering",
            "Information Technology",
            "Electrical and Electronics Engineering",
            "Electronics and Communication Engineering",
            "Mechanical Engineering",
            "general"
        ]
        
        # Check if response matches any valid course (case-insensitive)
        course_lower = course.lower()
        for valid_course in valid_courses:
            if valid_course.lower() in course_lower or course_lower in valid_course.lower():
                return valid_course
        
        # If no match, check for partial matches
        if "computer" in course_lower and "science" in course_lower:
            return "Computer Science and Engineering"
        elif "information" in course_lower and "technology" in course_lower:
            return "Information Technology"
        elif "electrical" in course_lower and "electronics" in course_lower:
            return "Electrical and Electronics Engineering"
        elif "electronics" in course_lower and "communication" in course_lower:
            return "Electronics and Communication Engineering"
        elif "mechanical" in course_lower and "engineering" in course_lower:
            return "Mechanical Engineering"
        else:
            return "general"
            
    except Exception as e:
        print(f"  ⚠ Error determining course: {e}, defaulting to 'general'")
        return "general"


def track_brochure_shared(phone_number: str, email: str, compiled_pdf_url: str, query: str):
    """Track brochure sharing in MongoDB brochuresShared collection"""
    if not mongodb_client:
        print("  ⚠ MongoDB client not available, skipping brochure tracking")
        return
    
    if not compiled_pdf_url:
        print("  ⚠ No PDF URL provided, skipping brochure tracking")
        return
    
    # At least one contact method must be provided
    if not phone_number and not email:
        print("  ⚠ No contact information provided, skipping brochure tracking")
        return
    
    try:
        # Normalize phone number to DB format (if provided)
        phone_db_format = None
        if phone_number:
            phone_db_format = normalize_phone_number(phone_number)
        
        # Determine course from query
        print(f"  🔍 Determining course from query...")
        course = determine_course_from_query(query)
        print(f"  ✓ Course determined: {course}")
        
        # Get database and collection
        db = mongodb_client["VELS"]
        brochures_collection = db["brochuresShared"]
        
        # Create document (only include fields that are provided)
        brochure_doc = {
            "compiled_pdf_url": compiled_pdf_url,
            "course": course
        }
        
        if phone_db_format:
            brochure_doc["phone_number"] = phone_db_format
        if email:
            brochure_doc["email"] = email
        
        # Insert into MongoDB
        result = brochures_collection.insert_one(brochure_doc)
        print(f"  ✓ Brochure sharing tracked in MongoDB (ID: {result.inserted_id})")
        
    except Exception as e:
        print(f"  ⚠ Error tracking brochure sharing: {e}")
        # Don't raise exception - this is a tracking feature, shouldn't break the main flow


def get_calendar_service():
    """Authenticate and return Google Calendar service"""
    if not GOOGLE_CALENDAR_AVAILABLE:
        raise ValueError("Google Calendar libraries not available")
    
    creds = None
    
    # First, try to load token from environment variable (base64 encoded)
    token_base64 = os.getenv('GOOGLE_CALENDAR_TOKEN')
    if token_base64:
        try:
            import base64
            token_data = base64.b64decode(token_base64)
            creds = pickle.loads(token_data)
            print("  ✓ Loaded calendar token from environment variable")
        except Exception as e:
            print(f"  ⚠ Failed to load token from environment: {e}")
            creds = None
    
    # If no token from env, try to load from pickle file (fallback)
    if not creds:
        token_path = os.path.join(os.path.dirname(__file__), 'token.pickle')
        if os.path.exists(token_path):
            try:
                with open(token_path, 'rb') as token:
                    creds = pickle.load(token)
                print("  ✓ Loaded calendar token from pickle file")
            except Exception as e:
                print(f"  ⚠ Failed to load token from pickle file: {e}")
                creds = None
    
    # If no valid credentials, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("  🔄 Refreshing expired token...")
            creds.refresh(Request())
            print("  ✓ Token refreshed successfully")
        else:
            # Load credentials from environment variable
            credentials_json = os.getenv('GOOGLE_CALENDAR_CREDENTIALS')
            
            if not credentials_json:
                raise ValueError("GOOGLE_CALENDAR_CREDENTIALS not found in .env file")
            
            # Parse JSON and create flow
            import json
            credentials_dict = json.loads(credentials_json)
            flow = InstalledAppFlow.from_client_config(
                credentials_dict, CALENDAR_SCOPES)
            print("  🔐 Starting OAuth flow...")
            creds = flow.run_local_server(port=0)
            print("  ✓ OAuth authentication completed")
        
        # Save credentials for future use (both to env format and pickle file)
        # Note: The base64 token should be updated in .env manually after first auth
        token_path = os.path.join(os.path.dirname(__file__), 'token.pickle')
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
        print("  ✓ Token saved to pickle file")
    
    return build('calendar', 'v3', credentials=creds)


def check_calendar_conflicts(date: str, start_time: str, end_time: str, timezone: str = 'Asia/Kolkata'):
    """
    Check if there are any events during the specified time period
    
    Args:
        date: Date in 'YYYY-MM-DD' format
        start_time: Time in 'HH:MM' format
        end_time: Time in 'HH:MM' format
        timezone: Timezone (default: 'Asia/Kolkata')
    
    Returns:
        List of conflicting events with details
    """
    try:
        service = get_calendar_service()
        
        # Build datetime strings in RFC3339 format
        # For Asia/Kolkata timezone, we need to add +05:30 offset
        time_min = f"{date}T{start_time}:00+05:30"
        time_max = f"{date}T{end_time}:00+05:30"
        
        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        if not events:
            return []
        
        # Format conflicting events for response
        conflicts = []
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            end = event['end'].get('dateTime', event['end'].get('date'))
            conflict_info = {
                'title': event.get('summary', 'Untitled Event'),
                'start': start,
                'end': end,
                'location': event.get('location', ''),
                'description': event.get('description', '')
            }
            conflicts.append(conflict_info)
        
        return conflicts
        
    except Exception as e:
        print(f"✗ Error checking conflicts: {e}")
        raise


def create_calendar_event(title: str, date: str, start_time: str, end_time: str, 
                         description: str = '', location: str = '', timezone: str = 'Asia/Kolkata'):
    """
    Create a calendar event
    
    Args:
        title: Event name
        date: Date in 'YYYY-MM-DD' format (e.g., '2024-12-01')
        start_time: Time in 'HH:MM' format (e.g., '14:00')
        end_time: Time in 'HH:MM' format (e.g., '15:00')
        description: Event description
        location: Event location
        timezone: Timezone (default: 'Asia/Kolkata')
    
    Returns:
        Created event object
    """
    try:
        service = get_calendar_service()
        
        # Build datetime strings
        start_datetime = f"{date}T{start_time}:00"
        end_datetime = f"{date}T{end_time}:00"
        
        event = {
            'summary': title,
            'location': location,
            'description': description,
            'start': {
                'dateTime': start_datetime,
                'timeZone': timezone,
            },
            'end': {
                'dateTime': end_datetime,
                'timeZone': timezone,
            },
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'popup', 'minutes': 30},
                    {'method': 'email', 'minutes': 1440},  # 24 hours before
                ],
            },
        }
        
        created_event = service.events().insert(calendarId='primary', body=event).execute()
        return created_event
        
    except Exception as e:
        print(f"✗ Error creating event: {e}")
        raise


def extract_text_with_ocr(page, page_num, pdf_name):
    """Extract text from a PDF page using OCR."""
    global TESSERACT_FOUND
    
    if not TESSERACT_FOUND:
        print(f"    ⚠ OCR skipped (Tesseract not available)")
        return ""
    
    try:
        print(f"    🔍 Running OCR on {pdf_name}&{page_num}...", end="", flush=True)
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(BytesIO(img_data))
        ocr_text = pytesseract.image_to_string(img, lang='eng')
        print(" ✓ Done")
        return ocr_text.strip()
    except pytesseract.TesseractNotFoundError:
        TESSERACT_FOUND = False
        print(" ✗ Failed (Tesseract not found)")
        return ""
    except Exception as e:
        print(f" ✗ Failed ({str(e)})")
        return ""


def extract_text_from_pdf(pdf_bytes, pdf_filename, use_ocr=True):
    """Extract text from PDF file page by page."""
    pdf_text = {}
    page_objects = {}  # Store page objects for S3 upload
    
    try:
        pdf_name = Path(pdf_filename).stem
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        print(f"\n{'='*80}")
        print(f"📄 Processing PDF: {pdf_filename}")
        print(f"📊 Total pages: {len(doc)}")
        print(f"🔧 OCR enabled: {use_ocr}")
        print(f"{'='*80}\n")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_identifier = f"{pdf_name}&{page_num + 1}"
            
            print(f"  📖 Page {page_num + 1}/{len(doc)} ({page_identifier})")
            
            print(f"    📝 Extracting regular text...", end="", flush=True)
            regular_text = page.get_text()
            regular_char_count = len(regular_text.strip())
            print(f" ✓ ({regular_char_count} chars)")
            
            ocr_text = ""
            ocr_char_count = 0
            if use_ocr:
                ocr_text = extract_text_with_ocr(page, page_num + 1, pdf_name)
                ocr_char_count = len(ocr_text.strip())
                if ocr_text:
                    print(f"      ✓ OCR extracted {ocr_char_count} chars")
            
            if regular_text.strip() and ocr_text.strip():
                if regular_text.strip() in ocr_text or len(regular_text.strip()) < 50:
                    combined_text = ocr_text
                    print(f"    💡 Using OCR text only (regular text contained in OCR)")
                else:
                    combined_text = f"{regular_text}\n\n[OCR Text from Images:]\n{ocr_text}"
                    print(f"    💡 Combined regular + OCR text")
            elif ocr_text.strip():
                combined_text = f"[OCR Text:]\n{ocr_text}"
                print(f"    💡 Using OCR text only")
            else:
                combined_text = regular_text
                print(f"    💡 Using regular text only")
            
            pdf_text[page_identifier] = combined_text
            page_objects[page_identifier] = page
            
            total_chars = len(combined_text.strip())
            print(f"    ✅ Page {page_identifier} complete - Total: {total_chars} chars")
            print(f"    {'-'*76}")
        
        print(f"\n{'='*80}")
        print(f"✅ PDF Processing Complete: {pdf_filename}")
        print(f"📊 Total pages processed: {len(pdf_text)}")
        print(f"{'='*80}\n")
        
        return pdf_text, page_objects, doc
        
    except Exception as e:
        print(f"\n❌ Error extracting text from {pdf_filename}: {str(e)}\n")
        raise Exception(f"Error extracting text from {pdf_filename}: {str(e)}")


def store_in_chromadb(page_identifier, text, pdf_name, page_number):
    """Store extracted text in ChromaDB with metadata. Prevents duplicates."""
    try:
        print(f"    💾 Storing {page_identifier} in ChromaDB...", end="", flush=True)
        
        # Check if this page already exists (prevent duplicates)
        existing = collection.get(
            where={"page_identifier": page_identifier},
            limit=1
        )
        
        if existing['ids']:
            # Page exists, update it instead of creating duplicate
            print(f" (updating existing)...", end="", flush=True)
            collection.update(
                ids=[existing['ids'][0]],
                documents=[text],
                metadatas=[{
                    "page_identifier": page_identifier,
                    "pdf_name": pdf_name,
                    "page_number": str(page_number),
                    "char_count": str(len(text))
                }]
            )
            print(" ✓ Updated")
        else:
            # New page, add it
            doc_id = f"{page_identifier}"
            collection.add(
                documents=[text],
                metadatas=[{
                    "page_identifier": page_identifier,
                    "pdf_name": pdf_name,
                    "page_number": str(page_number),
                    "char_count": str(len(text))
                }],
                ids=[doc_id]
            )
            print(" ✓ Stored")
        
        # Note: Backup is triggered after batch extraction completes, not after each page
        # This prevents race conditions and reduces S3 API calls
        
        return True
    except Exception as e:
        print(f" ✗ Failed ({str(e)})")
        return False


def upload_page_to_s3(page, page_identifier):
    """
    Convert PDF page to image and upload to S3.
    
    Args:
        page: PyMuPDF page object
        page_identifier: Page identifier (e.g., "E-Brochure-1&3")
    
    Returns:
        str: S3 URL of uploaded image, or None if failed
    """
    if s3_client is None:
        print(f"    ⚠ S3 client not configured, skipping upload for {page_identifier}")
        return None
    
    try:
        print(f"    📤 Uploading {page_identifier} to S3...", end="", flush=True)
        
        # Render page as high-quality image
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PNG bytes
        img_bytes = pix.tobytes("png")
        
        # S3 key (filename in bucket)
        s3_key = f"{page_identifier}.png"
        
        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=img_bytes,
            ContentType='image/png'
        )
        
        # Generate S3 URL
        s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
        
        print(f" ✓ Uploaded")
        return s3_url
        
    except ClientError as e:
        print(f" ✗ Failed (S3 Error: {e.response['Error']['Message']})")
        return None
    except Exception as e:
        print(f" ✗ Failed ({str(e)})")
        return None


def upload_full_pdf_to_s3(pdf_bytes: bytes, pdf_filename: str):
    """
    Upload the entire PDF file to S3.
    
    Args:
        pdf_bytes: PDF file contents as bytes
        pdf_filename: Original filename of the PDF
    
    Returns:
        str: S3 URL of uploaded PDF, or None if failed
    """
    if s3_client is None:
        print(f"    ⚠ S3 client not configured, skipping full PDF upload for {pdf_filename}")
        return None
    
    try:
        print(f"    📤 Uploading full PDF {pdf_filename} to S3...", end="", flush=True)
        
        # Sanitize filename for S3 key (remove special characters, keep extension)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = "".join(c for c in pdf_filename if c.isalnum() or c in ('-', '_', '.'))
        if not safe_filename.endswith('.pdf'):
            safe_filename = f"{safe_filename}.pdf"
        
        # S3 key (store in pdfs/ subfolder for organization)
        s3_key = f"pdfs/{timestamp}_{safe_filename}"
        
        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=pdf_bytes,
            ContentType='application/pdf',
            Metadata={
                'original_filename': pdf_filename,
                'uploaded_at': timestamp
            }
        )
        
        # Generate S3 URL
        s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
        
        print(f" ✓ Uploaded")
        return s3_url
        
    except ClientError as e:
        print(f" ✗ Failed (S3 Error: {e.response['Error']['Message']})")
        return None
    except Exception as e:
        print(f" ✗ Failed ({str(e)})")
        return None


def create_compiled_pdf_from_images(s3_image_urls, user_number, query):
    """
    Download images from S3, compile them into a single PDF, and upload to S3.
    
    Args:
        s3_image_urls: List of S3 image URLs
        user_number: User's phone number
        query: The original query
        
    Returns:
        str: S3 URL of compiled PDF, or None if failed
    """
    if s3_client is None:
        print(f"  ⚠ S3 client not configured, cannot create compiled PDF")
        return None
    
    try:
        print(f"  📄 Creating compiled PDF from {len(s3_image_urls)} images...")
        
        if not s3_image_urls:
            return None
        
        # Create a temporary PDF file
        import tempfile
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_pdf_path = temp_pdf.name
        temp_pdf.close()
        
        # Create PDF with ReportLab
        c = canvas.Canvas(temp_pdf_path, pagesize=A4)
        page_width, page_height = A4
        
        # Download and add each image to PDF
        for idx, img_url in enumerate(s3_image_urls, 1):
            print(f"    📥 Processing image {idx}/{len(s3_image_urls)}...", end="", flush=True)
            
            try:
                # Download image from S3
                response = requests.get(img_url, timeout=30)
                response.raise_for_status()
                
                # Open image with PIL
                img = Image.open(BytesIO(response.content))
                
                # Calculate dimensions to fit page while maintaining aspect ratio
                img_width, img_height = img.size
                aspect = img_height / float(img_width)
                
                # Fit to page with margins
                margin = 20
                available_width = page_width - (2 * margin)
                available_height = page_height - (2 * margin)
                
                if available_width * aspect <= available_height:
                    # Width is limiting factor
                    display_width = available_width
                    display_height = available_width * aspect
                else:
                    # Height is limiting factor
                    display_height = available_height
                    display_width = available_height / aspect
                
                # Center image on page
                x = (page_width - display_width) / 2
                y = (page_height - display_height) / 2
                
                # Draw image
                img_reader = ImageReader(BytesIO(response.content))
                c.drawImage(img_reader, x, y, width=display_width, height=display_height)
                
                # Add new page if not last image
                if idx < len(s3_image_urls):
                    c.showPage()
                
                print(" ✓")
                
            except Exception as e:
                print(f" ✗ Failed: {str(e)}")
                continue
        
        # Save PDF
        c.save()
        print(f"  ✓ PDF created successfully")
        
        # Upload to S3
        print(f"  📤 Uploading compiled PDF to S3...", end="", flush=True)
        
        # Generate filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"query_{user_number}_{timestamp}.pdf"
        
        # Read PDF file
        with open(temp_pdf_path, 'rb') as pdf_file:
            pdf_bytes = pdf_file.read()
        
        # Upload to S3 (using the same bucket as page images)
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=f"compiled_pdfs/{pdf_filename}",  # Store in a subfolder for organization
            Body=pdf_bytes,
            ContentType='application/pdf',
            Metadata={
                'user_number': user_number,
                'query': query[:200],  # Truncate long queries
                'page_count': str(len(s3_image_urls))
            }
        )
        
        # Generate direct S3 object URL (region-specific format)
        # Format: https://{bucket}.s3.{region}.amazonaws.com/{key}
        compiled_pdf_url = f"https://{S3_BUCKET_NAME}.s3.{aws_region}.amazonaws.com/compiled_pdfs/{pdf_filename}"
        print(f" ✓ Uploaded")
        print(f"  ✓ Compiled PDF URL: {compiled_pdf_url}")
        
        # Clean up temp file
        os.unlink(temp_pdf_path)
        
        return compiled_pdf_url
        
    except Exception as e:
        print(f"  ✗ Failed to create compiled PDF: {str(e)}")
        return None


def process_extraction_job(job_id: str, files_data: list, use_ocr: bool, webhook_url: Optional[str] = None):
    """
    Background function to process PDF extraction.
    
    Args:
        job_id: Unique job identifier
        files_data: List of tuples (filename, file_contents_bytes)
        use_ocr: Whether to use OCR
        webhook_url: Optional webhook URL to notify when complete
    """
    # Set extraction flag to prevent concurrent backups
    global extraction_in_progress
    with extraction_lock:
        extraction_in_progress = True
    
    try:
        with job_lock:
            extraction_jobs[job_id]["status"] = JobStatus.PROCESSING
            extraction_jobs[job_id]["started_at"] = datetime.now().isoformat()
        
        print("\n" + "="*80)
        print(f"🚀 Starting batch PDF extraction (Job ID: {job_id})")
        print(f"📁 Total files: {len(files_data)}")
        print(f"🔧 OCR enabled: {use_ocr}")
        print(f"🔧 OCR available: {TESSERACT_FOUND}")
        print("="*80)
        
        results = {}
        errors = []
        stored_count = 0
        s3_upload_count = 0
        pdf_s3_urls = {}  # Store full PDF S3 URLs by filename
        
        for idx, (filename, contents) in enumerate(files_data, 1):
            print(f"\n📦 Processing file {idx}/{len(files_data)}: {filename}")
            
            if not filename.lower().endswith('.pdf'):
                error_msg = f"{filename}: Not a PDF file"
                errors.append(error_msg)
                print(f"  ❌ {error_msg}")
                continue
            
            try:
                print(f"  📥 File size: {len(contents)} bytes")
                
                # Upload full PDF to S3 first
                full_pdf_s3_url = upload_full_pdf_to_s3(contents, filename)
                if full_pdf_s3_url:
                    pdf_s3_urls[filename] = full_pdf_s3_url
                
                # Extract text
                extracted_text, page_objects, doc = extract_text_from_pdf(contents, filename, use_ocr)
                
                # Store each page in ChromaDB and upload to S3
                pdf_name = Path(filename).stem
                print(f"\n  💾 Storing pages in ChromaDB and uploading to S3...")
                
                for page_identifier, text in extracted_text.items():
                    # Extract page number from identifier
                    page_number = int(page_identifier.split('&')[1])
                    
                    # Store in ChromaDB
                    if store_in_chromadb(page_identifier, text, pdf_name, page_number):
                        stored_count += 1
                    
                    # Upload page image to S3
                    page = page_objects[page_identifier]
                    s3_url = upload_page_to_s3(page, page_identifier)
                    if s3_url:
                        s3_upload_count += 1
                
                # Close the document
                doc.close()
                
                results.update(extracted_text)
                print(f"  ✅ Successfully extracted and stored {len(extracted_text)} pages from {filename}")
                
            except Exception as e:
                error_msg = f"{filename}: {str(e)}"
                errors.append(error_msg)
                print(f"  ❌ Failed: {error_msg}")
        
        # Backup ChromaDB to S3 after extraction completes
        if stored_count > 0 and s3_client is not None:
            print("📤 Backing up ChromaDB after extraction completes...")
            def delayed_backup():
                time.sleep(2)  # Wait 2 seconds for ChromaDB to flush writes
                upload_chromadb_to_s3()
            threading.Thread(target=delayed_backup, daemon=True).start()
        
        # Prepare response data
        response_data = {
            "job_id": job_id,
            "status": "success" if results else "failed",
            "total_files_processed": len(files_data),
            "total_pages_extracted": len(results),
            "total_pages_stored_in_db": stored_count,
            "total_pages_uploaded_to_s3": s3_upload_count,
            "ocr_enabled": use_ocr,
            "ocr_available": TESSERACT_FOUND,
            "page_identifiers": list(results.keys()),
            "pdf_s3_urls": pdf_s3_urls  # Full PDF S3 URLs
        }
        
        if errors:
            response_data["errors"] = errors
        
        # Update job status
        with job_lock:
            extraction_jobs[job_id]["status"] = JobStatus.COMPLETED if results else JobStatus.FAILED
            extraction_jobs[job_id]["completed_at"] = datetime.now().isoformat()
            extraction_jobs[job_id]["result"] = response_data
        
        print("\n" + "="*80)
        print(f"🎉 Batch extraction complete! (Job ID: {job_id})")
        print(f"✅ Success: {len(results)} pages extracted")
        print(f"💾 Stored: {stored_count} pages in ChromaDB")
        print(f"📤 Uploaded: {s3_upload_count} pages to S3")
        if errors:
            print(f"⚠ Errors: {len(errors)} files failed")
        print("="*80 + "\n")
        
        # Send webhook notification if provided
        if webhook_url:
            print(f"  🔔 Sending webhook notification to {webhook_url}...")
            send_webhook_notification(webhook_url, job_id, response_data)
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Job {job_id} failed: {error_msg}\n")
        
        # Update job status to failed
        with job_lock:
            extraction_jobs[job_id]["status"] = JobStatus.FAILED
            extraction_jobs[job_id]["completed_at"] = datetime.now().isoformat()
            extraction_jobs[job_id]["error"] = error_msg
            extraction_jobs[job_id]["result"] = {
                "job_id": job_id,
                "status": "failed",
                "error": error_msg
            }
        
        # Send webhook notification for failure
        if webhook_url:
            print(f"  🔔 Sending failure webhook notification to {webhook_url}...")
            failure_data = {
                "job_id": job_id,
                "status": "failed",
                "error": error_msg
            }
            send_webhook_notification(webhook_url, job_id, failure_data)
    finally:
        # Always reset extraction flag when done
        with extraction_lock:
            extraction_in_progress = False


@app.post("/extract")
async def extract_text(
    files: List[UploadFile] = File(...),
    use_ocr: bool = Form(True),
    webhook_url: Optional[str] = Form(None)
):
    """
    Extract text from multiple PDF files and store in ChromaDB.
    
    If webhook_url is provided, the extraction will be processed asynchronously
    and a webhook will be called when complete. Otherwise, it processes synchronously.
    
    Args:
        files: List of PDF files to extract
        use_ocr: Whether to use OCR for text extraction
        webhook_url: Optional webhook URL to notify when extraction is complete
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # If webhook_url is provided, process asynchronously
    if webhook_url:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Read all file contents into memory (required for async processing)
        files_data = []
        for file in files:
            contents = await file.read()
            files_data.append((file.filename, contents))
        
        # Initialize job tracking
        with job_lock:
            extraction_jobs[job_id] = {
                "job_id": job_id,
                "status": JobStatus.PENDING,
                "created_at": datetime.now().isoformat(),
                "total_files": len(files),
                "use_ocr": use_ocr,
                "webhook_url": webhook_url,
                "result": None,
                "error": None
            }
        
        # Start background processing
        print(f"\n🔄 Starting async extraction job: {job_id}")
        print(f"📡 Webhook URL: {webhook_url}")
        threading.Thread(
            target=process_extraction_job,
            args=(job_id, files_data, use_ocr, webhook_url),
            daemon=True
        ).start()
        
        # Return immediately with job ID
        return JSONResponse(content={
            "status": "processing",
            "job_id": job_id,
            "message": "Extraction started. You will be notified via webhook when complete.",
            "webhook_url": webhook_url,
            "check_status_url": f"/extract/status/{job_id}"
        })
    
    # Synchronous processing (original behavior when no webhook)
    # Set extraction flag to prevent concurrent backups
    global extraction_in_progress
    with extraction_lock:
        extraction_in_progress = True
    
    try:
        print("\n" + "="*80)
        print(f"🚀 Starting batch PDF extraction (synchronous)")
        print(f"📁 Total files received: {len(files)}")
        print(f"🔧 OCR enabled: {use_ocr}")
        print(f"🔧 OCR available: {TESSERACT_FOUND}")
        print("="*80)
        
        results = {}
        errors = []
        stored_count = 0
        s3_upload_count = 0
        pdf_s3_urls = {}  # Store full PDF S3 URLs by filename
        
        for idx, file in enumerate(files, 1):
            print(f"\n📦 Processing file {idx}/{len(files)}: {file.filename}")
            
            if not file.filename.lower().endswith('.pdf'):
                error_msg = f"{file.filename}: Not a PDF file"
                errors.append(error_msg)
                print(f"  ❌ {error_msg}")
                continue
            
            try:
                print(f"  📥 Reading file contents...")
                contents = await file.read()
                print(f"  ✓ File size: {len(contents)} bytes")
        
                # Upload full PDF to S3 first
                full_pdf_s3_url = upload_full_pdf_to_s3(contents, file.filename)
                if full_pdf_s3_url:
                    pdf_s3_urls[file.filename] = full_pdf_s3_url
        
                # Extract text
                extracted_text, page_objects, doc = extract_text_from_pdf(contents, file.filename, use_ocr)
                
                # Store each page in ChromaDB and upload to S3
                pdf_name = Path(file.filename).stem
                print(f"\n  💾 Storing pages in ChromaDB and uploading to S3...")
                
                for page_identifier, text in extracted_text.items():
                    # Extract page number from identifier
                    page_number = int(page_identifier.split('&')[1])
                    
                    # Store in ChromaDB
                    if store_in_chromadb(page_identifier, text, pdf_name, page_number):
                        stored_count += 1
                    
                    # Upload page image to S3
                    page = page_objects[page_identifier]
                    s3_url = upload_page_to_s3(page, page_identifier)
                    if s3_url:
                        s3_upload_count += 1
                
                # Close the document
                doc.close()
                
                results.update(extracted_text)
                print(f"  ✅ Successfully extracted and stored {len(extracted_text)} pages from {file.filename}")
                
            except Exception as e:
                error_msg = f"{file.filename}: {str(e)}"
                errors.append(error_msg)
                print(f"  ❌ Failed: {error_msg}")
        
        response = {
            "status": "success" if results else "failed",
            "total_files_processed": len(files),
            "total_pages_extracted": len(results),
            "total_pages_stored_in_db": stored_count,
            "total_pages_uploaded_to_s3": s3_upload_count,
            "ocr_enabled": use_ocr,
            "ocr_available": TESSERACT_FOUND,
            "page_identifiers": list(results.keys()),
            "pdf_s3_urls": pdf_s3_urls  # Full PDF S3 URLs
        }
        
        if errors:
            response["errors"] = errors
        
        print("\n" + "="*80)
        print(f"🎉 Batch extraction complete!")
        print(f"✅ Success: {len(results)} pages extracted")
        print(f"💾 Stored: {stored_count} pages in ChromaDB")
        print(f"📤 Uploaded: {s3_upload_count} pages to S3")
        if errors:
            print(f"⚠ Errors: {len(errors)} files failed")
        print("="*80 + "\n")
    
        # Backup ChromaDB to S3 after extraction completes (ensures all pages are written)
        if stored_count > 0 and s3_client is not None:
            print("📤 Backing up ChromaDB after extraction completes...")
            # Use a small delay to ensure all ChromaDB writes are flushed to disk
            def delayed_backup():
                time.sleep(2)  # Wait 2 seconds for ChromaDB to flush writes
                upload_chromadb_to_s3()
            threading.Thread(target=delayed_backup, daemon=True).start()
        
        return JSONResponse(content=response)
    finally:
        # Always reset extraction flag when done
        with extraction_lock:
            extraction_in_progress = False


@app.get("/extract/status/{job_id}")
async def get_extraction_status(job_id: str):
    """
    Get the status of an extraction job.
    
    Args:
        job_id: The job identifier returned from the /extract endpoint
    
    Returns:
        Job status and result (if completed)
    """
    with job_lock:
        if job_id not in extraction_jobs:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )
        
        job = extraction_jobs[job_id]
        
        response = {
            "job_id": job_id,
            "status": job["status"].value,
            "created_at": job["created_at"],
            "total_files": job["total_files"],
            "use_ocr": job["use_ocr"],
            "webhook_url": job.get("webhook_url")
        }
        
        if "started_at" in job:
            response["started_at"] = job["started_at"]
        
        if "completed_at" in job:
            response["completed_at"] = job["completed_at"]
        
        if job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED]:
            response["result"] = job["result"]
            if job.get("error"):
                response["error"] = job["error"]
        
        return JSONResponse(content=response)


@app.post("/query")
async def query_documents(request: QueryRequest):
    """
    Query the stored PDF documents using GPT-4o, ChromaDB, and send results via WhatsApp and Email.
    """
    query = request.query
    user_number = request.number
    user_email = request.email
    
    # Validate that at least one contact method is provided
    if not user_number and not user_email:
        raise HTTPException(
            status_code=400,
            detail="At least one of 'number' or 'email' must be provided"
        )
    
    print("\n" + "="*80)
    print(f"🔍 Processing query: {query}")
    print(f"📱 User number: {user_number if user_number else 'Not provided'}")
    print(f"📧 User email: {user_email if user_email else 'Not provided'}")
    print("="*80)
    
    try:
        # Query ChromaDB for relevant documents - increase results to ensure we get good matches
        print(f"  📊 Searching ChromaDB for relevant documents...")
        
        # Extract PDF name from query to filter results to relevant PDF only
        query_lower = query.lower()
        pdf_name_filter = None
        
        # Common property name patterns and their PDF identifiers
        # Map common property name variations to their PDF file names
        # Order matters: longer/more specific patterns should be checked first
        pdf_name_mappings = [
            ('palm premiere', 'palm-premiere-brochure'),
            ('palmpremiere', 'palm-premiere-brochure'),
            ('palmpremier', 'palm-premiere-brochure'),
        ]
        
        # Try to detect PDF name from query (check for property names, longest first)
        for property_name, pdf_name in pdf_name_mappings:
            if property_name in query_lower:
                pdf_name_filter = pdf_name
                print(f"  🔍 Detected property: '{property_name}' → Filtering to PDF: {pdf_name}")
                break
        
        # If no specific property detected, check all stored PDFs to infer
        if not pdf_name_filter:
            # Get all unique PDF names from collection
            all_results = collection.get(limit=1000)
            unique_pdf_names = set()
            if all_results.get('metadatas'):
                for meta in all_results['metadatas']:
                    if meta.get('pdf_name'):
                        unique_pdf_names.add(meta['pdf_name'])
            
            # Try fuzzy match on query text
            for stored_pdf in unique_pdf_names:
                pdf_lower = stored_pdf.lower().replace('-', ' ').replace('_', ' ')
                # Check if any significant words from PDF name appear in query
                pdf_words = pdf_lower.split()
                query_words = query_lower.split()
                if any(word in query_words for word in pdf_words if len(word) > 4):
                    pdf_name_filter = stored_pdf
                    print(f"  🔍 Matched query to PDF: {pdf_name_filter}")
                    break
        
        # Enhanced search query - add keywords that might help find relevant content
        enhanced_query = query
        if 'kitchen' in query_lower:
            enhanced_query = f"{query} kitchen specifications details features"
        elif 'bedroom' in query_lower:
            enhanced_query = f"{query} bedroom specifications details features"
        elif 'bathroom' in query_lower:
            enhanced_query = f"{query} bathroom specifications details features"
        
        # Build query with optional PDF filter
        query_params = {
            'query_texts': [enhanced_query],
            'n_results': 20  # Get more results to ensure we find relevant pages
        }
        
        # Add PDF filter if detected
        if pdf_name_filter:
            query_params['where'] = {'pdf_name': pdf_name_filter}
            print(f"  ✓ Filtering to PDF: {pdf_name_filter}")
        else:
            print(f"  ⚠ No specific PDF detected, searching all PDFs")
        
        results = collection.query(**query_params)
        
        if not results['documents'] or len(results['documents'][0]) == 0:
            print(f"  ⚠ No documents found in ChromaDB")
            return JSONResponse(content={
                "status": "no_results",
                "summary": "No relevant documents found in the database.",
                "pages": [],
                "s3_images": [],
                "compiled_pdf_url": None,
                "whatsapp_status": {"status": "skipped", "reason": "No results"},
                "email_status": {"status": "skipped", "reason": "No results"}
            })
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        # Deduplicate by page_identifier (keep best relevance score)
        seen_pages = {}
        deduplicated_docs = []
        deduplicated_metas = []
        deduplicated_distances = []
        
        for doc, meta, dist in zip(documents, metadatas, distances):
            page_id = meta['page_identifier']
            relevance = 1 - dist
            
            # Keep only the best relevance score for each page
            if page_id not in seen_pages or relevance > seen_pages[page_id]['relevance']:
                seen_pages[page_id] = {
                    'doc': doc,
                    'meta': meta,
                    'dist': dist,
                    'relevance': relevance
                }
        
        # Sort by relevance (highest first) and take top results
        sorted_pages = sorted(seen_pages.items(), key=lambda x: x[1]['relevance'], reverse=True)
        
        # Take top 15 unique pages
        for page_id, page_data in sorted_pages[:15]:
            deduplicated_docs.append(page_data['doc'])
            deduplicated_metas.append(page_data['meta'])
            deduplicated_distances.append(page_data['dist'])
        
        print(f"  ✓ Found {len(deduplicated_docs)} unique relevant documents (from {len(documents)} total results)")
        
        # Prepare context for GPT-4o
        context_parts = []
        pages_used = []
        
        for i, (doc, metadata, distance) in enumerate(zip(deduplicated_docs, deduplicated_metas, deduplicated_distances)):
            page_identifier = metadata['page_identifier']
            pages_used.append(page_identifier)
            context_parts.append(f"[Source: {page_identifier}]\n{doc}\n")
            relevance_score = 1 - distance
            print(f"    📄 {page_identifier} (relevance: {relevance_score:.3f})")
        
        context = "\n".join(context_parts)
        
        # Check if OpenAI client is available
        if openai_client is None:
            print(f"  ❌ OpenAI client not configured. Cannot process query.")
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured. Please set OPENAI_API_KEY in .env file."
            )
        
        # Query GPT-4o with structured output
        print(f"\n  🤖 Querying GPT-4o...")
        
        system_prompt = """You are a helpful assistant that answers questions based on PDF document content.
You MUST use the provided document excerpts to answer the user's query. 
Your response MUST be in JSON format with exactly two fields:
1. "summary": A comprehensive answer based ONLY on the provided document excerpts
2. "pages_used": An array of page identifiers (e.g., ["palm-premiere-brochure&19", "palm-premiere-brochure&14"]) that you used from the provided excerpts

CRITICAL RULES:
- You MUST include ALL page identifiers that contain information relevant to answering the query
- Even if information appears in multiple pages, include all those page identifiers
- If the document excerpts contain relevant information, you MUST use it and list the source pages
- Only exclude pages that have NO relevant information at all
- Be thorough - if kitchen details are in page 19, you MUST include "palm-premiere-brochure&19" in pages_used"""

        user_prompt = f"""Query: "{query}"

Below are document excerpts from a real estate brochure. Answer the query using ONLY information from these excerpts. Be comprehensive and include all relevant details.

Document excerpts:
{context}

REQUIREMENTS:
1. Extract ALL relevant information from the excerpts to answer: "{query}"
2. List ALL page identifiers that contain information relevant to your answer
3. Be thorough - include pages even if they only have partial information

Respond in JSON format:
{{
  "summary": "Your detailed answer based on the excerpts above. Include specific details, measurements, features mentioned.",
  "pages_used": ["page-identifier-1", "page-identifier-2", ...]
}}"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent, factual responses
            max_tokens=2000,  # Increased for more detailed answers
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        import json
        result = json.loads(response.choices[0].message.content)
        summary = result.get("summary", "")
        pages_actually_used = result.get("pages_used", [])
        
        # Ensure pages_actually_used are valid (exist in pages_used)
        pages_actually_used = [p for p in pages_actually_used if p in pages_used]
        
        # If GPT-4o returns no pages, return no results (no fallback)
        if not pages_actually_used:
            print(f"  ⚠ GPT-4o returned 0 pages - no relevant documents found")
            print(f"  → Returning no results (no fallback)")
            return JSONResponse(content={
                "status": "no_results",
                "summary": "No relevant documents found in the database for this query.",
                "pages": [],
                "s3_images": [],
                "compiled_pdf_url": None,
                "whatsapp_status": {"status": "skipped", "reason": "No results"},
                "email_status": {"status": "skipped", "reason": "No results"}
            })
        
        print(f"  ✓ GPT-4o response generated")
        print(f"  📄 GPT-4o used {len(pages_actually_used)} out of {len(pages_used)} retrieved pages")
        
        # Log which pages were actually used
        for page in pages_actually_used:
            print(f"    ✓ Used: {page}")
        
        # Fetch S3 URLs for the pages used
        s3_images = []
        if s3_client is not None:
            print(f"\n  🔗 Fetching S3 image URLs...")
            for page_identifier in pages_actually_used:
                s3_key = f"{page_identifier}.png"
                s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
                
                # Verify if the object exists in S3 (optional but recommended)
                try:
                    s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                    s3_images.append(s3_url)
                    print(f"    ✓ Found: {s3_url}")
                except ClientError:
                    print(f"    ⚠ Not found in S3: {page_identifier}")
        else:
            print(f"\n  ⚠ S3 client not configured, skipping S3 URL fetching")
        
        # Create compiled PDF from images
        compiled_pdf_url = None
        # Use number or email for PDF filename (prefer number, fallback to email hash)
        pdf_identifier = user_number if user_number else (user_email.split('@')[0] if user_email else "unknown")
        if s3_client is not None and s3_images:
            print(f"\n  📚 Creating compiled PDF...")
            compiled_pdf_url = create_compiled_pdf_from_images(s3_images, pdf_identifier, query)
        else:
            print(f"\n  ⚠ Skipping compiled PDF creation (S3 not configured or no images)")
        
        # Start background thread for WhatsApp and Email (non-blocking)
        if user_number or user_email:
            print(f"\n  🔄 Starting background thread for message sending...")
            background_thread = threading.Thread(
                target=send_messages_background,
                args=(user_number, user_email, summary, compiled_pdf_url, query),
                daemon=True
            )
            background_thread.start()
            print(f"  ✓ Background thread started (messages will be sent asynchronously)")
        
        print(f"\n{'='*80}")
        print(f"✅ Query completed successfully")
        print(f"📄 Pages retrieved: {len(pages_used)}, Pages used: {len(pages_actually_used)}")
        print(f"🖼️  S3 images found: {len(s3_images)}")
        print(f"📚 Compiled PDF: {'Created' if compiled_pdf_url else 'Failed'}")
        print(f"📱 WhatsApp: Processing in background")
        print(f"📧 Email: Processing in background")
        print(f"{'='*80}\n")
        
        # Return response immediately (don't wait for WhatsApp/Email)
        return JSONResponse(content={
            "status": "success",
            "summary": summary,
            "pages": pages_actually_used,
            "s3_images": s3_images,
            "compiled_pdf_url": compiled_pdf_url,
            "whatsapp_status": {"status": "processing", "message": "WhatsApp message is being sent in the background"},
            "email_status": {"status": "processing", "message": "Email is being sent in the background"}
        })
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/calendar")
async def create_calendar_event_endpoint(request: CalendarRequest):
    """
    Create a calendar event with conflict checking.
    If there are overlapping events, returns an error with conflict details.
    Only creates the event if there are no conflicts.
    """
    if not GOOGLE_CALENDAR_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="Google Calendar libraries not available. Please install required packages."
        )
    
    print("\n" + "="*80)
    print(f"📅 Creating calendar event")
    print(f"  Title: {request.title}")
    print(f"  Date: {request.date}")
    print(f"  Time: {request.start_time} - {request.end_time}")
    print(f"  Location: {request.location}")
    print("="*80)
    
    try:
        # First, check for conflicts
        print(f"  🔍 Checking for conflicts...")
        conflicts = check_calendar_conflicts(
            date=request.date,
            start_time=request.start_time,
            end_time=request.end_time
        )
        
        if conflicts:
            print(f"  ⚠ Found {len(conflicts)} conflicting event(s)")
            conflict_details = []
            for conflict in conflicts:
                conflict_details.append({
                    "title": conflict['title'],
                    "start": conflict['start'],
                    "end": conflict['end'],
                    "location": conflict.get('location', '')
                })
                print(f"    - {conflict['title']} ({conflict['start']} to {conflict['end']})")
            
            return JSONResponse(
                status_code=409,  # Conflict status code
                content={
                    "status": "conflict",
                    "message": f"There are {len(conflicts)} overlapping event(s) at this time. Please choose a different time slot.",
                    "conflicts": conflict_details,
                    "requested_time": {
                        "date": request.date,
                        "start_time": request.start_time,
                        "end_time": request.end_time
                    }
                }
            )
        
        # No conflicts, proceed to create the event
        print(f"  ✓ No conflicts found, creating event...")
        created_event = create_calendar_event(
            title=request.title,
            date=request.date,
            start_time=request.start_time,
            end_time=request.end_time,
            description=request.description,
            location=request.location
        )
        
        print(f"  ✓ Event created successfully!")
        print(f"    → Link: {created_event.get('htmlLink')}")
        print(f"    → Event ID: {created_event.get('id')}")
        
        return JSONResponse(content={
            "status": "success",
            "message": "Event created successfully",
            "event": {
                "id": created_event.get('id'),
                "title": created_event.get('summary'),
                "start": created_event['start'].get('dateTime', created_event['start'].get('date')),
                "end": created_event['end'].get('dateTime', created_event['end'].get('date')),
                "location": created_event.get('location', ''),
                "htmlLink": created_event.get('htmlLink')
            }
        })
        
    except ValueError as e:
        print(f"  ❌ Error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create calendar event: {str(e)}"
        )


@app.get("/pdfs")
async def list_pdfs():
    """
    List all PDFs stored in ChromaDB with their metadata.
    Returns information about each PDF including name, page count, and total characters.
    """
    try:
        print("\n" + "="*80)
        print(f"📋 Listing all PDFs in ChromaDB...")
        print("="*80)
        
        # Get all documents from ChromaDB
        # Use a large limit to get all documents, or paginate if needed
        all_results = collection.get(limit=10000)  # Adjust limit as needed
        
        if not all_results.get('metadatas') or len(all_results['metadatas']) == 0:
            print("  ⚠ No PDFs found in ChromaDB")
            return JSONResponse(content={
                "status": "success",
                "total_pdfs": 0,
                "pdfs": []
            })
        
        # Group by pdf_name
        pdf_info = {}
        
        for metadata in all_results['metadatas']:
            pdf_name = metadata.get('pdf_name', 'unknown')
            page_number = int(metadata.get('page_number', 0))
            char_count = int(metadata.get('char_count', 0))
            
            if pdf_name not in pdf_info:
                pdf_info[pdf_name] = {
                    "pdf_name": pdf_name,
                    "page_count": 0,
                    "total_characters": 0,
                    "pages": []
                }
            
            pdf_info[pdf_name]["page_count"] += 1
            pdf_info[pdf_name]["total_characters"] += char_count
            pdf_info[pdf_name]["pages"].append({
                "page_number": page_number,
                "page_identifier": metadata.get('page_identifier', ''),
                "characters": char_count
            })
        
        # Sort pages by page number for each PDF
        for pdf_name in pdf_info:
            pdf_info[pdf_name]["pages"].sort(key=lambda x: x["page_number"])
        
        # Convert to list and sort by PDF name
        pdfs_list = list(pdf_info.values())
        pdfs_list.sort(key=lambda x: x["pdf_name"])
        
        print(f"  ✓ Found {len(pdfs_list)} PDF(s) in ChromaDB")
        for pdf in pdfs_list:
            print(f"    📄 {pdf['pdf_name']}: {pdf['page_count']} pages, {pdf['total_characters']:,} characters")
        
        print("="*80 + "\n")
        
        return JSONResponse(content={
            "status": "success",
            "total_pdfs": len(pdfs_list),
            "pdfs": pdfs_list
        })
        
    except Exception as e:
        print(f"  ❌ Error listing PDFs: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Failed to list PDFs: {str(e)}")


@app.delete("/pdfs/{pdf_name}")
async def delete_pdf(pdf_name: str, delete_s3_images: bool = False):
    """
    Delete a PDF and all its pages from ChromaDB.
    
    Args:
        pdf_name: The name of the PDF to delete (as stored in ChromaDB metadata)
        delete_s3_images: If True, also delete corresponding S3 images (default: False)
    
    Returns:
        Information about what was deleted
    """
    try:
        print("\n" + "="*80)
        print(f"🗑️  Deleting PDF: {pdf_name}")
        print(f"   Delete S3 images: {delete_s3_images}")
        print("="*80)
        
        # First, get all pages for this PDF to see what we're deleting
        pdf_results = collection.get(
            where={"pdf_name": pdf_name},
            limit=10000
        )
        
        if not pdf_results.get('ids') or len(pdf_results['ids']) == 0:
            print(f"  ⚠ PDF '{pdf_name}' not found in ChromaDB")
            raise HTTPException(
                status_code=404,
                detail=f"PDF '{pdf_name}' not found in ChromaDB"
            )
        
        page_count = len(pdf_results['ids'])
        page_identifiers = []
        
        if pdf_results.get('metadatas'):
            for metadata in pdf_results['metadatas']:
                page_id = metadata.get('page_identifier', '')
                if page_id:
                    page_identifiers.append(page_id)
        
        print(f"  📊 Found {page_count} page(s) to delete")
        
        # Delete from ChromaDB using where clause (deletes all matching documents)
        print(f"  🗑️  Deleting from ChromaDB...", end="", flush=True)
        try:
            collection.delete(
                where={"pdf_name": pdf_name}
            )
            print(" ✓ Deleted")
        except Exception as delete_error:
            # Fallback: delete by IDs if where clause doesn't work
            print(f" ⚠ Delete by where failed, trying by IDs...", end="", flush=True)
            collection.delete(ids=pdf_results['ids'])
            print(" ✓ Deleted")
        
        # Optionally delete S3 images
        s3_deleted_count = 0
        s3_deleted_keys = []
        if delete_s3_images and s3_client is not None:
            print(f"  🗑️  Deleting S3 images...")
            for page_identifier in page_identifiers:
                s3_key = f"{page_identifier}.png"
                try:
                    s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                    s3_deleted_count += 1
                    s3_deleted_keys.append(s3_key)
                    print(f"    ✓ Deleted: {s3_key}")
                except ClientError as e:
                    print(f"    ⚠ Failed to delete {s3_key}: {e.response['Error']['Message']}")
                except Exception as e:
                    print(f"    ⚠ Failed to delete {s3_key}: {str(e)}")
        elif delete_s3_images and s3_client is None:
            print(f"  ⚠ S3 client not configured, skipping S3 image deletion")
        
        # Delete full PDF files from S3 (always delete if S3 is configured)
        pdf_deleted_count = 0
        pdf_deleted_keys = []
        if s3_client is not None:
            print(f"  🗑️  Deleting full PDF files from S3...")
            try:
                # List all objects in the pdfs/ prefix
                paginator = s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix='pdfs/')
                
                # Create variations of pdf_name to match against
                # pdf_name is the stem (filename without extension) from ChromaDB
                # We need to match against the sanitized filename in S3
                pdf_name_lower = pdf_name.lower()
                
                # Find and delete matching PDFs
                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            s3_key = obj['Key']
                            
                            # Extract filename from S3 key (format: pdfs/{timestamp}_{filename})
                            # Remove 'pdfs/' prefix and timestamp prefix
                            if s3_key.startswith('pdfs/'):
                                filename_part = s3_key[5:]  # Remove 'pdfs/'
                                # Remove timestamp prefix (format: YYYYMMDD_HHMMSS_)
                                # Find first underscore after potential timestamp
                                parts = filename_part.split('_', 2)
                                if len(parts) >= 3:
                                    # Likely has timestamp, get the filename part
                                    potential_filename = '_'.join(parts[2:])
                                else:
                                    # No timestamp or different format, use whole thing
                                    potential_filename = filename_part
                                
                                # Get filename without extension for comparison
                                potential_name_stem = Path(potential_filename).stem.lower()
                                
                                # Check if this PDF matches our pdf_name
                                if pdf_name_lower == potential_name_stem or pdf_name_lower in potential_filename.lower():
                                    try:
                                        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                                        pdf_deleted_count += 1
                                        pdf_deleted_keys.append(s3_key)
                                        print(f"    ✓ Deleted full PDF: {s3_key}")
                                    except ClientError as e:
                                        print(f"    ⚠ Failed to delete {s3_key}: {e.response['Error']['Message']}")
                                    except Exception as e:
                                        print(f"    ⚠ Failed to delete {s3_key}: {str(e)}")
                
                if pdf_deleted_count == 0:
                    print(f"    ⚠ No matching full PDF files found in S3 for '{pdf_name}'")
            except ClientError as e:
                print(f"    ⚠ Error listing/deleting PDFs from S3: {e.response['Error']['Message']}")
            except Exception as e:
                print(f"    ⚠ Error deleting PDFs from S3: {str(e)}")
        else:
            print(f"  ⚠ S3 client not configured, skipping full PDF deletion")
        
        # Trigger backup after deletion
        if s3_client is not None:
            print(f"  📤 Backing up ChromaDB after deletion...")
            def delayed_backup():
                time.sleep(2)  # Wait for ChromaDB to flush writes
                upload_chromadb_to_s3()
            threading.Thread(target=delayed_backup, daemon=True).start()
        
        print(f"\n{'='*80}")
        print(f"✅ PDF deletion completed")
        print(f"   PDF: {pdf_name}")
        print(f"   Pages deleted from ChromaDB: {page_count}")
        if delete_s3_images:
            print(f"   S3 images deleted: {s3_deleted_count}")
        print(f"   Full PDFs deleted from S3: {pdf_deleted_count}")
        print(f"{'='*80}\n")
        
        return JSONResponse(content={
            "status": "success",
            "message": f"PDF '{pdf_name}' deleted successfully",
            "pdf_name": pdf_name,
            "pages_deleted_from_chromadb": page_count,
            "s3_images_deleted": s3_deleted_count if delete_s3_images else None,
            "s3_image_keys_deleted": s3_deleted_keys if delete_s3_images else None,
            "full_pdfs_deleted_from_s3": pdf_deleted_count,
            "full_pdf_keys_deleted": pdf_deleted_keys
        })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"  ❌ Error deleting PDF: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Failed to delete PDF: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Check for required API keys and credentials
    print("\n" + "="*80)
    print("🔧 Configuration Check")
    print("="*80)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY not found!")
    else:
        print("✓ OpenAI API key configured")
    
    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        print("⚠️  WARNING: AWS credentials not found!")
    else:
        print("✓ AWS credentials configured")
    
    if not os.getenv("TWILIO_ACCOUNT_SID") or not os.getenv("TWILIO_AUTH_TOKEN"):
        print("⚠️  WARNING: Twilio credentials not found!")
    else:
        print("✓ Twilio credentials configured")
    
    if not os.getenv("SMTP_USERNAME") or not os.getenv("SMTP_PASSWORD"):
        print("⚠️  WARNING: SMTP credentials not found!")
    else:
        print("✓ SMTP credentials configured")
    
    if not GOOGLE_CALENDAR_AVAILABLE:
        print("⚠️  WARNING: Google Calendar libraries not available!")
    elif not os.getenv("GOOGLE_CALENDAR_CREDENTIALS"):
        print("⚠️  WARNING: GOOGLE_CALENDAR_CREDENTIALS not found!")
    else:
        print("✓ Google Calendar credentials configured")
    
    print("\nPlease ensure your .env file contains:")
    print("  OPENAI_API_KEY=your-openai-api-key")
    print("  AWS_ACCESS_KEY_ID=your-aws-access-key")
    print("  AWS_SECRET_ACCESS_KEY=your-aws-secret-key")
    print("  AWS_REGION=us-east-1")
    print("  S3_BUCKET_NAME=your-bucket-name")
    print("  TWILIO_ACCOUNT_SID=your-twilio-account-sid")
    print("  TWILIO_AUTH_TOKEN=your-twilio-auth-token")
    print("  TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886")
    print("  SMTP_SERVER=smtp.gmail.com (or your SMTP server)")
    print("  SMTP_PORT=587")
    print("  SMTP_USERNAME=your-email@gmail.com")
    print("  SMTP_PASSWORD=your-app-password")
    print("  SMTP_FROM_EMAIL=your-email@gmail.com (optional, defaults to SMTP_USERNAME)")
    print("  GOOGLE_CALENDAR_CREDENTIALS=your-google-calendar-credentials-json")
    print("  GOOGLE_CALENDAR_TOKEN=base64-encoded-token-pickle (optional, for pre-authenticated tokens)")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8080)
