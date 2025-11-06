import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def avaliar_modelo(nome, modelo, X_val, y_val, X_te, y_te):
    yv = modelo.predict(X_val)
    yt = modelo.predict(X_te)

    # Se o modelo retornar valores contínuos, converter para classes ±1
    if np.any((yv != -1) & (yv != 1)):
        yv = np.sign(yv)
        yv[yv == 0] = 1  # tratar o caso raro de 0
    if np.any((yt != -1) & (yt != 1)):
        yt = np.sign(yt)
        yt[yt == 0] = 1

    print(f"\n=== {nome} | Treinamento ===")
    print(confusion_matrix(y_val, yv, labels=[-1, 1]))
    print(classification_report(y_val, yv, labels=[-1, 1], target_names=["Dígito 5 (-1)", "Dígito 1 (+1)"]))
    print("Acurácia (val): {:.2f}%".format(100*accuracy_score(y_val, yv)))

    print(f"\n=== {nome} | Teste ===")
    print(confusion_matrix(y_te, yt, labels=[-1, 1]))
    print(classification_report(y_te, yt, labels=[-1, 1],target_names=["Dígito 5 (-1)", "Dígito 1 (+1)"]))
    print("Acurácia (test): {:.2f}%".format(100*accuracy_score(y_te, yt)))