def calcular_intensidade(imagem):
    soma = 0.0
    for i in range(len(imagem)):
        soma += imagem[i]
    intensidade = soma / 255.0
    return intensidade

def calcular_simetria(imagem):
    # Reconstruir a matriz 28x28 a partir do vetor
    matriz = []
    k = 0
    for i in range(28):
        linha = []
        for j in range(28):
            linha.append(imagem[k])
            k += 1
        matriz.append(linha)

    # Simetria vertical (comparando colunas espelhadas)
    sim_v = 0.0
    for i in range(28):
        for j in range(14):
            # pixel da esquerda vs pixel espelhado da direita
            valor_esq = matriz[i][j]
            valor_dir = matriz[i][27 - j]
            sim_v += abs(valor_esq - valor_dir)

    # Simetria horizontal (comparando linhas espelhadas)
    sim_h = 0.0
    for i in range(14):
        for j in range(28):
            valor_cima = matriz[i][j]
            valor_baixo = matriz[27 - i][j]
            sim_h += abs(valor_cima - valor_baixo)

    # Normaliza o total (como no original)
    sim_total = (sim_v + sim_h) / 255.0
    return sim_total