from PIL import Image, ImageDraw, ImageFont

import random

width, height = 500, 500

for i in range(10):
    r = lambda: random.randint(0,255)
    color = '#%02X%02X%02X' % (r(),r(),r())
    img = Image.new(mode="RGBA", size=(500, 500), color=color)
    font = ImageFont.truetype("arial.ttf", 400)
    draw = ImageDraw.Draw(img)
    
    text = f"{i}"
    w, h = draw.textsize(text, font=font)
    h += int(h * 0.21)
    
    draw.text(((width- w) / 2, (height - h) / 2), text=text, fill="white", font=font)
    
    img.save(f'image_{i}.png')