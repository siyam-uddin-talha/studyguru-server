import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.core.config import settings
from app.helpers.email_templates import (
    generate_otp_email,
    generate_reset_pin_email,
)
from jinja2 import Environment, FileSystemLoader
import os
from email.utils import formataddr

# Setup Jinja2 environment
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
env = Environment(loader=FileSystemLoader(templates_dir))


def render_template(template_name: str, **kwargs):
    template = env.get_template(template_name)
    return template.render(**kwargs)


async def send_email(to_email: str, subject: str, html_content: str):
    """Send email using SMTP"""
    try:
        msg = MIMEMultipart()
        app_name = settings.APP_NAME
        from_email = settings.SMTP_EMAIL_ADDRESS  # e.g. info@studyguru.pro

        msg["From"] = formataddr((app_name, from_email))

        msg["To"] = to_email
        msg["Subject"] = subject

        msg.attach(MIMEText(html_content, "html"))

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


async def send_welcome_email(email: str, recipient_name: str, start_link: str):
    """Send Welcome email"""
    subject = "Welcome to " + settings.APP_NAME
    html_content = render_template(
        "welcome_email.html", name=recipient_name, email=email, start_link=start_link
    )
    return await send_email(email, subject, html_content)
