import unicodedata

def padronizar_resultado(texto):
    """
    Remove acentos e deixa o texto minúsculo, trocando espaços por underline.
    """
    texto = unicodedata.normalize('NFD', texto)
    texto = texto.encode('ascii', 'ignore').decode('utf-8')
    texto = texto.lower().replace(' ', '_')
    return texto

def padronizar_nome_arquivo(nome):
    """
    Padroniza o nome de arquivos removendo acentos, colocando tudo em minúsculas e trocando espaços e hifens por underscore.
    """
    nome = unicodedata.normalize('NFD', nome)
    nome = nome.encode('ascii', 'ignore').decode('utf-8')
    nome = nome.lower().replace(' ', '_').replace('-', '_')
    return nome
