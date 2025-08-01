def generate_otp_email(otp: str, recipient_name: str) -> str:
    """Generate OTP verification email template"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Email Verification</title>
    </head>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            <h2 style="color: #4CAF50;">Email Verification</h2>
            <p>Hello {recipient_name},</p>
            <p>Thank you for registering with Inner States Therapy. Please use the following verification code to complete your registration:</p>
            <div style="background-color: #f4f4f4; padding: 20px; text-align: center; margin: 20px 0;">
                <h1 style="color: #4CAF50; font-size: 32px; margin: 0;">{otp}</h1>
            </div>
            <p>This code will expire in 15 minutes.</p>
            <p>If you didn't request this verification, please ignore this email.</p>
            <p>Best regards,<br>Inner States Therapy Team</p>
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
    </head>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            <h2 style="color: #FF6B6B;">Password Reset</h2>
            <p>Hello {recipient_name},</p>
            <p>You requested to reset your password. Please use the following code to reset your password:</p>
            <div style="background-color: #f4f4f4; padding: 20px; text-align: center; margin: 20px 0;">
                <h1 style="color: #FF6B6B; font-size: 32px; margin: 0;">{otp}</h1>
            </div>
            <p>This code will expire in 15 minutes.</p>
            <p>If you didn't request this password reset, please ignore this email.</p>
            <p>Best regards,<br>Inner States Therapy Team</p>
        </div>
    </body>
    </html>
    """


def generate_welcome_email(recipient_name: str, recipient_email: str, quick_start_link: str) -> str:
    """Generate welcome email template"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Welcome to Inner States Therapy</title>
    </head>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            <h2 style="color: #4CAF50;">Welcome to Inner States Therapy!</h2>
            <p>Hello {recipient_name},</p>
            <p>Welcome to Inner States Therapy! We're excited to have you join our community.</p>
            <p>Your account has been successfully created with the email: <strong>{recipient_email}</strong></p>
            <div style="text-align: center; margin: 30px 0;">
                <a href="{quick_start_link}" style="background-color: #4CAF50; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block;">Get Started</a>
            </div>
            <p>If you have any questions, feel free to reach out to our support team.</p>
            <p>Best regards,<br>Inner States Therapy Team</p>
        </div>
    </body>
    </html>
    """