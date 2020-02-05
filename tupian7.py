from PIL import Image
img = Image.open("图片4.png")

for i in range(209):
    for j in range(226):
        try:
            r, g, b, alpha = img.getpixel((i, j))
            if (r <= 120 and r >= 0) and (g <= 190 and g >= 125) and (b <= 70 and b >= 30):
                r = 0;g = 176;b = 240;img.putpixel((i, j), (r, g, b, alpha))

            if (r <= 245 and r >= 120) and (g <= 110 and g >= 0) and (b <= 85 and b >= 0):
                r = 112; g = 48; b = 160; img.putpixel((i, j), (r, g, b, alpha))
        except Exception as e:
            continue
img.show()