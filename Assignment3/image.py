from PIL import Image, ImageFilter
from scipy import signal as sg
import numpy as np
import matplotlib.pyplot as plt

def out(conv, path, filename):
    '''
    Normalizes pixel values and saves image.

    Arguments 
        conv: convoluted image to save
        path: image path
        filename: image filename
    '''
    norm = 255. * np.absolute(conv) / np.max(conv)
    out = Image.fromarray(norm.round().astype(np.uint8), 'L')
    out.save(path + filename)

def sobel(img):
    '''
    Implements the Sobel operator to calculate the gradients.

    Arguments
        img: input image

    Returns
        convoluted image
    '''
    gx = np.array([[1., 0., -1.], 
                [2., 0., -2.], 
                [1., 0., -1.]])

    gy = np.array([[1., 2., 1.],
                [0., 0., 0.],
                [-1., -2., -1.]])

    convx = sg.convolve(img, gx)
    convy = sg.convolve(img, gy)
    return np.sqrt(convx**2 + convy**2)

def scharr(img):
    '''
    Implements the Scharr operator to calculate the gradients.

    Arguments
        img: input image

    Returns
        convoluted image
    '''
    gx = np.array([[3., 0., -3.],
                [10., 0., -10.],
                [3., 0., -3.]])

    gy = np.array([[3., 10., 3.],
                [0., 0., 0.],
                [-3., -10., -3.]])

    convx = sg.convolve(img, gx)
    convy = sg.convolve(img, gy)
    return np.sqrt(convx**2 + convy**2)

def roberts(img):
    '''
    Implements the Roberts operator to calculate the gradients

    Arguments
        img: input image

    Returns
        convoluted image
    '''
    # Roberts operator
    gx = np.array([[1., 0.],
                [0., -1.]])

    gy = np.array([[0., 1.],
                [-1., 0.]])

    convx = sg.convolve(img, gx)
    convy = sg.convolve(img, gy)
    return np.sqrt(convx**2 + convy**2)

def prewitt(img):
    '''
    Implements the Prewitt operator to calculate the gradients.

    Arguments
        img: input image

    Returns
        convoluted image
    '''
    gx = np.array([[1., 0., -1.],
                [1., 0., -1.],
                [1., 0., -1.]])

    gy = np.array([[1., 1., 1.],
                [0., 0., 0.],
                [-1., -1., -1.]])

    convx = sg.convolve(img, gx)
    convy = sg.convolve(img, gy)
    return np.sqrt(convx**2 + convy**2)

def gaussian(img, sigma):
    '''
    Implements the first derivative of a Gaussian to calculate the gradients.

    Arguments
        img: input image
        sigma: sigma value for the Gaussian

    Returns
        convoluted image
    '''
    x, y = np.mgrid[-3:4, -3:4]
    g_arr = -2.*(x+y) / sigma * np.exp(-(x**2+y**2)/sigma) 
    
    convx = sg.convolve(img, g_arr)
    return convx

def change_blur(img, name, rad, kernel):
    '''
    Finds edges in an image for a range of different gaussian blur radii. Saves 
    images to a file.

    Arguments
        img: input image
        name: filename to save series of images to
        rad: list of radius values
        kernel: type of kernel to convolve with
    '''
    for r in rad:
        g_blur = img.filter(ImageFilter.GaussianBlur(r)).convert('L')
        #bfunc = brightness(gauss)
        arr = np.asarray(g_blur, dtype=np.float32)

        if kernel == 'sobel':
            output = sobel(arr)

        elif kernel == 'scharr':
            output = scharr(arr)

        elif kernel == 'roberts':
            output = roberts(arr)
        
        elif kernel == 'gaussian':
            output = gaussian(arr, 1.)

        else:
            output = prewitt(arr)
        
        out(output, 'edges/blur_radius/{}/'.format(name), '{}_r={}.jpg'.format(kernel, r))

def change_sigma(img, name, sigmas, r):
    '''
    Finds edges in an image for a range of different sigma values for the Gaussian
    gradient kernel. Saves images to a file.

    Arguments
        img: input image
        name: filename to save series of images to
        sigmas: list of sigma values
        r: Gaussian blur radius
    '''
    for sigma in sigmas:
        g_blur = img.filter(ImageFilter.GaussianBlur(r)).convert('L')
        arr = np.asarray(g_blur, dtype=np.float32)
        output = gaussian(arr, sigma)
        out(output, 'edges/sigmas/{}/'.format(name), 'sigma={}.jpg'.format(sigma))

def find_edge(img, name, r):
    '''
    Finds edges in an image using different kernels. Saves images to a file.

    Arguments
        r: gaussian blur radius
        img: input image
        name: filename to save series of images to
    '''
    blur = img.filter(ImageFilter.GaussianBlur(r)).convert('L')
    #bfunc = brightness(gauss)
    arr = np.asarray(blur, dtype=np.float32)
    
    sob = sobel(arr)
    sch = scharr(arr)
    rob = roberts(arr)
    pre = prewitt(arr)
    gauss = gaussian(arr, 1.)
    
    out(sob, 'edges/kernels/{}/'.format(name), 'sobel.jpg')
    out(sch, 'edges/kernels/{}/'.format(name), 'scharr.jpg')
    out(rob, 'edges/kernels/{}/'.format(name), 'roberts.jpg')
    out(pre, 'edges/kernels/{}/'.format(name), 'prewitt.jpg')   
    out(gauss, 'edges/kernels/{}/'.format(name), 'gaussian.jpg')


# run
starwars = Image.open('images/starwars.jpg')
bears = Image.open('images/bears.jpg')
bird = Image.open('images/bird.jpg')
tigers = Image.open('images/tigers.jpg')

print(starwars.filename, starwars.size)
print(bears.filename, bears.size)
print(bird.filename, bird.size)
print(tigers.filename, tigers.size)

#Test FIND_EDGES
edges = bird.filter(ImageFilter.FIND_EDGES)
edges.show()

sigmas = [0.1, 0.5, 1., 3., 5.]
blurs = [0, 0.5, 1, 5, 7]

find_edge(starwars, 'starwars', 0.2)
find_edge(bird, 'bird', 0.2)
find_edge(bears, 'bears', 0.2)
find_edge(tigers, 'tigers', 0.2)

change_blur(starwars, 'starwars',  blurs, 'sobel')
change_blur(bird, 'bird', blurs, 'sobel')
change_blur(bears, 'bears', blurs, 'sobel')
change_blur(tigers, 'tigers', blurs, 'sobel')

change_sigma(starwars, 'starwars', sigmas, 0.2)
change_sigma(bird, 'bird', sigmas, 0.2)
change_sigma(bears, 'bears', sigmas, 0.2)
change_sigma(tigers, 'tigers', sigmas, 0.2)
