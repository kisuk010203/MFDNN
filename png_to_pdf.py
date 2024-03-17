from PIL import Image
from reportlab.pdfgen import canvas
import sys


def convert_png_to_pdf(png_path):
    pdf_path = png_path.split(".")[0] + ".pdf"
    # Open the image file
    img = Image.open(png_path)
    # If the image has an alpha channel, convert it to white background
    if img.mode == "RGBA":
        alpha = Image.new("L", img.size, 255)
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=alpha)
        img = bg
    # Create a new PDF file
    c = canvas.Canvas(pdf_path, pagesize=img.size)
    # Draw the image on the PDF
    c.drawImage(png_path, 0, 0, *img.size)
    # Save the PDF
    c.save()



# Usage
png_path = sys.argv[1]
convert_png_to_pdf(png_path)
