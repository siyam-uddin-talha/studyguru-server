import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.core.config import settings
from app.helpers.email_templates import generate_otp_email, generate_reset_pin_email


async def send_email(to_email: str, subject: str, html_content: str):
    """Send email using SMTP"""
    try:
        msg = MIMEMultipart()
        msg['From'] = settings.SMTP_EMAIL_ADDRESS
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(html_content, 'html'))
        
        server = smtplib.SMTP(settings.SMTP_EMAIL_HOST, settings.SMTP_EMAIL_PORT)
        server.starttls()
        server.login(settings.SMTP_EMAIL_ADDRESS, settings.SMTP_EMAIL_PASSWORD)
        
        text = msg.as_string()
        server.sendmail(settings.SMTP_EMAIL_ADDRESS, to_email, text)
        server.quit()
        
        return True
    except Exception as e:
        print(f"Email sending failed: {e}")
        return False


async def send_verification_email(email: str, pin: int, recipient_name: str):
    """Send verification email with OTP"""
    subject = "Verify Your Email"
    html_content = generate_otp_email(str(pin), recipient_name)
    return await send_email(email, subject, html_content)


async def send_reset_email(email: str, pin: int, recipient_name: str):
    """Send password reset email with PIN"""
    subject = "Use this OTP to reset your password"
    html_content = generate_reset_pin_email(str(pin), recipient_name)
    return await send_email(email, subject, html_content)