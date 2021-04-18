import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

def add_noise(noise_typ, img):
    """Add noise with numpy."""
    image = img / 255.
    if noise_typ == "gauss":
        row, col, ch= image.shape
        mean = 0
        var = 0.002
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        out = gauss + image
        out =  np.clip(out, 0, 1)
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.02
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * row * col * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape[:2]]
        out[tuple(coords)] = np.random.uniform(0.6, 1.0)
        # Pepper mode
        num_pepper = np.ceil(amount* row * col * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape[:2]]
        out[tuple(coords)] = np.random.uniform(0, 0.4)
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        out = np.clip(noisy, 0, 1)
    return (out * 255).astype(np.uint8)


def apply_filter(filter_type, img):
    """Apply blur or color filter."""
    image = Image.fromarray(img)
    if filter_type == 'blur':
        out = image.filter(filter=ImageFilter.BLUR)
    elif filter_type == 'color':
        enhancer = ImageEnhance.Color(image)
        out = enhancer.enhance(np.random.uniform(0.4, 1.3))
    elif filter_type == 'gblur':
        out = image.filter(ImageFilter.GaussianBlur(1))
    return np.array(out)


def random_transformation(image):
    """
    Transform image with random filter or noise.
    Image assumed to be an array of shape (h, w, ch) 
    and color values are in [0, 255] range.
    Always varies colors saluration, and add noise/blur with 60% propability.
    """
    transformations = [
        lambda x: add_noise('gauss', x),
        lambda x: add_noise('s&p', x),
        lambda x: add_noise('poisson', x),
        lambda x: apply_filter('blur', x),
        lambda x: apply_filter('gblur', x),
        lambda x: x, # no change
    ]
    p = [0.12, 0.12, 0.12, 0.12, 0.12, 0.40]
    
    trans_fn = np.random.choice(transformations, p=p)
    return apply_filter('color', trans_fn(image.astype(np.uint8))).astype(image.dtype)


if __name__ == '__main__':
    with Image.open('dataset/train/happy/0.jpg') as image:
        img = np.array(image.convert("RGB"))
        Image.fromarray(random_transformation(img)).show()