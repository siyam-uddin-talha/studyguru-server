def generate_otp_email(otp: str, recipient_name: str) -> str:
    """Generate OTP verification email template"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Email Verification</title>
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
                line-height: 1.6; 
                color: #1F1F1F; 
                background-color: #FFFFFF; 
                max-width: 600px; 
                margin: 0 auto; 
                padding: 20px;
            }}
            .container {{
                background-color: #FFFFFF;
                border: 1px solid #E9E9E9;
                border-radius: 8px;
                padding: 30px;
            }}
            .verification-code {{
                background-color: #F5F5F5;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
                margin: 20px 0;
            }}
            .verification-code h1 {{
                color: #000000;
                font-size: 32px;
                margin: 0;
                letter-spacing: 2px;
            }}
            .footer {{
                color: #6F6F6F;
                font-size: 14px;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Email Verification</h2>
            <p>Hello {recipient_name},</p>
            <p>Thank you for registering with Study Guru - Pro. Please use the following verification code to complete your registration:</p>
            <div class="verification-code">
                <h1>{otp}</h1>
            </div>
            <p>This code will expire in 15 minutes.</p>
            <p>If you didn't request this verification, please ignore this email.</p>
            <div class="footer">
                <p>Best regards,<br>Study Guru - Pro Team</p>
            </div>
        </div>
    </body>
    </html>
    """

def generate_reset_pin_email(otp: str, recipient_name: str) -> str:
    """Generate password reset email template"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Password Reset</title>
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
                line-height: 1.6; 
                color: #1F1F1F; 
                background-color: #FFFFFF; 
                max-width: 600px; 
                margin: 0 auto; 
                padding: 20px;
            }}
            .container {{
                background-color: #FFFFFF;
                border: 1px solid #E9E9E9;
                border-radius: 8px;
                padding: 30px;
            }}
            .verification-code {{
                background-color: #F5F5F5;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
                margin: 20px 0;
            }}
            .verification-code h1 {{
                color: #000000;
                font-size: 32px;
                margin: 0;
                letter-spacing: 2px;
            }}
            .footer {{
                color: #6F6F6F;
                font-size: 14px;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Password Reset</h2>
            <p>Hello {recipient_name},</p>
            <p>You requested to reset your password. Please use the following code to reset your password:</p>
            <div class="verification-code">
                <h1>{otp}</h1>
            </div>
            <p>This code will expire in 15 minutes.</p>
            <p>If you didn't request this password reset, please ignore this email.</p>
            <div class="footer">
                <p>Best regards,<br>Study Guru - Pro Team</p>
            </div>
        </div>
    </body>
    </html>
    """

def generate_welcome_email(
    recipient_name: str, recipient_email: str, quick_start_link: str
) -> str:
    """Generate welcome email template"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Welcome to Study Guru - Pro</title>
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
                line-height: 1.6; 
                color: #1F1F1F; 
                background-color: #FFFFFF; 
                max-width: 600px; 
                margin: 0 auto; 
                padding: 20px;
            }}
            .container {{
                background-color: #FFFFFF;
                border: 1px solid #E9E9E9;
                border-radius: 8px;
                padding: 30px;
            }}
            .cta-button {{
                display: inline-block;
                background-color: #000000;
                color: #FFFFFF !important;
                text-decoration: none;
                padding: 12px 24px;
                border-radius: 6px;
                margin: 20px 0;
                text-align: center;
            }}
            .footer {{
                color: #6F6F6F;
                font-size: 14px;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Welcome to Study Guru - Pro!</h2>
            <p>Hello {recipient_name},</p>
            <p>Welcome to Study Guru - Pro! We're excited to have you join our community.</p>
            <p>Your account has been successfully created with the email: <strong>{recipient_email}</strong></p>
            <div style="text-align: center;">
                <a href="{quick_start_link}" class="cta-button">Get Started</a>
            </div>
            <p>If you have any questions, feel free to reach out to our support team.</p>
            <div class="footer">
                <p>Best regards,<br>Study Guru - Pro Team</p>
            </div>
        </div>
    </body>
    </html>
    """
