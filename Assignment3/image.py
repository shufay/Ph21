from PIL import Image, ImageFilter
im = Image.open('starwars')

im_edge = im.filter(ImageFilter.FIND_EDGES)
print(im.format, im.size, im.mode)
im_edge.show()
