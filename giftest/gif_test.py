import PIL.Image as Image


names = ['seed=0.png','seed=1.png','seed=2.png','seed=3.png']
imgs = []

for name in names:

    img = Image.open(name)
    imgs.append(img)


imgs[0].save('out.gif', save_all=True, append_images=imgs, optimize=True, duration=500, loop=0)