from PIL import Image
import imagehash

def get_image_hash(image_path):
    img = Image.open(image_path).convert("RGB")
    return str(imagehash.phash(img))