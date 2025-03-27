import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import relatif des identifiants
from config.code import EMAIL_ADDRESS, EMAIL_APP_PASSWORD

def send_error_email(subject, message, to_email="nicedevia@gmail.com"):
    print("📧 Envoi du mail en cours...")
    from_email = EMAIL_ADDRESS
    app_password = EMAIL_APP_PASSWORD

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(message, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, app_password)
            server.send_message(msg)
        print("✅ Email envoyé avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors de l'envoi du mail : {e}")
