"""
Contact form handling routes.
Handles Trust Code programme form submissions and sends notifications.
"""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

from fastapi import APIRouter, HTTPException  # type: ignore
from pydantic import BaseModel, EmailStr  # type: ignore

# Configure logging
_LOG = logging.getLogger(__name__)

router = APIRouter(prefix="/contact", tags=["contact"])


class TrustCodeFormSubmission(BaseModel):
    organizationName: str
    website: str
    country: str
    vertical: str
    audienceSize: str
    contactName: str
    email: EmailStr
    timeframe: str
    acknowledgeTerms: bool


def _send_email(subject: str, body: str,
                to_email: str = "denis@nirvana.bm") -> bool:
    """
    Send email using SMTP.
    Returns True if successful, False otherwise.
    """
    try:
        # Get email configuration from environment variables
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        if not smtp_user or not smtp_password:
            _LOG.error("SMTP credentials not configured")
            return False

        from_email = os.getenv("FROM_EMAIL") or smtp_user

        # Create message
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject

        # Add body to email
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        # Create SMTP session
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Enable TLS encryption
        server.login(smtp_user, smtp_password)

        # Send email
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()

        _LOG.info(f"Email sent successfully to {to_email}")
        return True

    except Exception as e:
        _LOG.error(f"Failed to send email: {str(e)}")
        return False


def _format_trust_code_email(
        form_data: TrustCodeFormSubmission) -> tuple[str, str]:
    """
    Format the Trust Code form submission into email subject and body.
    Returns (subject, body) tuple.
    """
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    subject = f"Trust Code Programme Request - {form_data.organizationName}"

    terms_ack = "Yes" if form_data.acknowledgeTerms else "No"
    body = f"""
New Trust Code Programme Request

Submission Time: {timestamp}

Organization Details:
- Organization Name: {form_data.organizationName}
- Website: {form_data.website}
- Country: {form_data.country}
- Vertical: {form_data.vertical}
- Audience Size: {form_data.audienceSize}

Contact Information:
- Contact Name: {form_data.contactName}
- Email: {form_data.email}
- Preferred Go-live Timeframe: {form_data.timeframe}

Terms Acknowledgment: {terms_ack}

---
This request was submitted through the Trust Code Programme page at
nirvana.bm/trust-code-programme
"""

    return subject, body


@router.post("/trust-code-request")
def submit_trust_code_request(form_data: TrustCodeFormSubmission) -> dict:
    """
    Handle Trust Code programme form submission.
    Validates the data and sends notification email.
    """
    try:
        # Validate required fields
        if not form_data.acknowledgeTerms:
            raise HTTPException(
                status_code=400,
                detail="Terms acknowledgment is required"
            )

        # Format email content
        subject, body = _format_trust_code_email(form_data)

        # Send notification email
        email_sent = _send_email(subject, body)

        if not email_sent:
            # Log the form data for manual processing if email fails
            _LOG.warning(
                f"Email delivery failed for Trust Code request: "
                f"org={form_data.organizationName}, "
                f"contact={form_data.contactName}, "
                f"email={form_data.email}"
            )
            # Still return success to user - we have the data logged

        return {
            "success": True,
            "message": "Thank you for your interest! We will be in touch soon."
        }

    except HTTPException:
        raise
    except Exception as e:
        _LOG.error(f"Error processing Trust Code request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred processing your request. "
                   "Please try again later."
        )


@router.get("/health")
def contact_health_check() -> dict:
    """Health check endpoint for contact services."""
    return {"status": "ok", "service": "contact"}
