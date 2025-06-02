from PIL import Image
from rembg import remove
import io

def remove_background_and_center(image: Image.Image) -> Image.Image:
    """
    Remove fundo da imagem com U²-Net (via rembg), centraliza o objeto,
    aplica fundo branco e redimensiona para 256x256.
    """
    # Remove fundo usando rembg
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()

    output_bytes = remove(image_bytes)
    image_rgba = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    # Redimensionar mantendo proporção
    image_rgba.thumbnail((256, 256), Image.LANCZOS)

    # Criar imagem 256x256 com fundo branco
    background = Image.new("RGBA", (256, 256), (255, 255, 255, 255))
    paste_x = (256 - image_rgba.width) // 2
    paste_y = (256 - image_rgba.height) // 2
    background.paste(image_rgba, (paste_x, paste_y), image_rgba)

    return background.convert("RGB")
