import qrcode
from PIL import Image

chatbot_url = "https://myfirstap.streamlit.app/?view=chatbot"  # Replace with your actual URL
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data(chatbot_url)
qr.make(fit=True)
img = qr.make_image(fill='black', back_color='white')

# Save or display the QR code image
img.save("chatbot_qr.png")
