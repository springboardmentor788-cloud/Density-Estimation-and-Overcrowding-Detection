import smtplib
from email.mime.text import MIMEText
import datetime

# Dummy Config (Replace with real credentials)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_app_password"
RECEIVER_EMAIL = "admin@example.com"

def send_email_alert(current_count, threshold):
    """
    Sends an email alert when the crowd density exceeds threshold.
    """
    print(f"[ALERT] Triggering email alert! Count {current_count} exceeds {threshold}.")
    
    # The following snippet demonstrates how an SMTP script operates.
    # To execute safely in local testing, it stops short of actually logging in to standard ports via dummy passwords.
    
    # We write a local log to simulate it.
    with open("outputs/alerts.log", "a") as f:
        f.write(f"[{datetime.datetime.now()}] ALERT ISSUED: Overcrowding detected (Count: {current_count} / Threshold: {threshold})\n")
    
    """
    try:
        msg = MIMEText(f"Overcrowding alert triggered at {datetime.datetime.now()}.\n"
                       f"Current count: {current_count}\nThreshold: {threshold}.")
        msg['Subject'] = '🚨 ALERT: Overcrowding Detected!'
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print("Alert email sent successfully.")
    except Exception as e:
        print(f"Failed to send email alert: {e}")
    """
