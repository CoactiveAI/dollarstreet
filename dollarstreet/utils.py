from PIL import Image


def pil_loader(path: str) -> Image.Image:
    img = Image.open(path)
    return img.convert("RGB")
