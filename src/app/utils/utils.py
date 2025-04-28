import unicodedata

def padronizar_resultado(texto):
    """
    Remove acentos e deixa o texto minúsculo, trocando espaços por underline.
    """
    texto = unicodedata.normalize('NFD', texto)
    texto = texto.encode('ascii', 'ignore').decode('utf-8')
    texto = texto.lower().replace(' ', '_')
    return texto
