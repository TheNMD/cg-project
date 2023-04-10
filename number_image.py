from PIL import Image, ImageDraw, ImageFont

width, height = 500, 500

for i in range(10):
    img = Image.new("L", (500, 500), color=0)
    font = ImageFont.truetype("arial.ttf", 400)
    draw = ImageDraw.Draw(img)
    
    text = f"{i}"
    w, h = draw.textsize(text, font=font)
    h += int(h*0.21)
    
    draw.text(((width- w) / 2, (height - h) / 2), text=text, fill='white', font=font)
    
    img.save(f'image_{i}.png')