import smtplib

EMAIL_SENDER = "surekha17012002@gmail.com"
EMAIL_PASSWORD = "wujx wice wlnx gmsx"

try:
    print("📤 Connecting to SMTP server...")
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()

    print("🔑 Logging in...")
    server.login(EMAIL_SENDER, EMAIL_PASSWORD)
    print("✅ Login successful!")

    server.quit()
except Exception as e:
    print(f"❌ Email login failed: {e}")
