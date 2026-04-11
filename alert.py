import smtplib
from email.message import EmailMessage
import os
import threading

try:
    from twilio.rest import Client
except ImportError:
    Client = None

def send_alert_async(count):
    """Starts the alert process in a background thread."""
    thread = threading.Thread(target=_send_alert_task, args=(count,), daemon=True)
    thread.start()

def _send_alert_task(count):
    """Actual alert logic supporting both SMTP and Twilio, running in background."""
    print(f"🚨 Overcrowding detected! Count: {count}")
    
    # Send Email
    try:
        sender = os.getenv("ALERT_SENDER_EMAIL", "kalyanvardhan037@gmail.com")
        password = os.getenv("ALERT_APP_PASSWORD", "Kalyan@123")
        receiver = os.getenv("ALERT_RECEIVER_EMAIL", "praneethsingh0606@gmail.com")
        
        if sender and password and receiver:
            msg = EmailMessage()
            msg.set_content(f"🚨 Overcrowding detected!\nEstimated Count: {count}")
            msg['Subject'] = '🚨 Crowd Alert'
            msg['From'] = sender
            msg['To'] = receiver

            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(sender, password)
                server.send_message(msg)

            print("✅ Email Alert Sent")
        else:
            print("⚠️ Email credentials not fully configured. Set ALERT_SENDER_EMAIL, ALERT_APP_PASSWORD, ALERT_RECEIVER_EMAIL in environment.")
    except Exception as e:
        print(f"❌ Email Failed: {e}")

    # Send SMS via Twilio if configured
    twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_auth = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_from = os.getenv("TWILIO_PHONE_NUMBER")
    twilio_to = os.getenv("TWILIO_TO_NUMBER")

    if twilio_sid and twilio_auth and twilio_from and twilio_to and Client:
        try:
            client = Client(twilio_sid, twilio_auth)
            message = client.messages.create(
                body=f"🚨 Overcrowding Alert! Crowd size reached {int(count)}.",
                from_=twilio_from,
                to=twilio_to
            )
            print(f"✅ Twilio SMS Sent: SID {message.sid}")
        except Exception as e:
            print(f"❌ Twilio Failed: {e}")
    elif twilio_sid:
        print("⚠️ Twilio library missing or configuration incomplete. Install twilio via pip.")