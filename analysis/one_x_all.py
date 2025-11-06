import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score


# === Função para treinar um contra todos ===
def treinar_um_contra_todos(classificador, X_train, y_train, X_test, y_test, digitos_alvo=[0,1,4,5]):
    """
    Treina classificadores um-contra-todos (One-vs-Rest) para cada dígito em 'digitos_alvo',
    mas o último dígito é considerado 'default' (sem classificador).
    """
    X_train_rest = X_train.copy()
    y_train_rest = y_train.copy()

    classificadores = {}
    scalers = {}

    # Treina para todos os dígitos, menos o último
    for digito in digitos_alvo[:-1]:
        y_bin = np.where(y_train_rest == digito, 1, -1)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train_rest)

        clf = classificador()
        clf.execute(X_scaled, y_bin)

        classificadores[digito] = clf
        scalers[digito] = scaler

        pred_train = clf.predict(X_scaled)
        mask_rest = pred_train != 1
        X_train_rest = X_train_rest[mask_rest]
        y_train_rest = y_train_rest[mask_rest]

        if len(X_train_rest) == 0:
            print(f"Todas as amostras foram classificadas antes de treinar o dígito {digito}.")
            break

    # Predição no teste
    previsoes = []
    for x in X_test:
        pred = None
        for digito in digitos_alvo[:-1]:
            clf = classificadores[digito]
            scaler = scalers[digito]
            x_scaled = scaler.transform([x])
            if clf.predict(x_scaled)[0] == 1:
                pred = digito
                break
        # O último dígito é o default
        if pred is None:
            pred = digitos_alvo[-1]
        previsoes.append(pred)

    print(f"\n=== Modelo: {classificador.__name__} ===")
    print(classification_report(y_test, previsoes, digits=2))

    return {digito: (classificadores[digito], scalers[digito]) for digito in classificadores}, previsoes


# === Função auxiliar para calcular a fronteira ===
def calcula_y_model(model, x_vals):
    """
    Calcula a coordenada y da fronteira de decisão (reta) considerando o bias.
    Funciona para qualquer modelo com atributo 'w'.
    """
    w = model.w
    # Caso típico: w = [bias, w1, w2]
    if len(w) >= 3 and not np.isclose(w[2], 0):
        return (-w[0] - w[1]*x_vals) / w[2]
    else:
        return np.full_like(x_vals, -w[0]/w[1])  # linha vertical


# === Função para plotar fronteiras um contra todos ===
def plot_um_contra_todos(classificadores_scalers, X_original, y_original, digitos_alvo=[0, 1, 4, 5]):
    cores_pontos = {0: 'purple', 1: 'lightgreen', 4: 'salmon', 5: 'violet'}
    marcadores = {0: 'o', 1: 's', 4: '^', 5: 'D'}
    cores_linhas = {0: 'purple', 1: 'lightgreen', 4: 'salmon', 5: 'violet'}
    estilos_linhas = {0: '--', 1: '-.', 4: ':', 5: '-'}

    plt.figure(figsize=(10, 8))

    # --- Plot dos pontos ---
    for digito in np.unique(y_original):
        mask = y_original == digito
        plt.scatter(X_original[mask, 0], X_original[mask, 1],
                    c=cores_pontos.get(digito, 'gray'),
                    marker=marcadores.get(digito, 'o'),
                    edgecolor='k', alpha=0.6,
                    label=f'Dígito {digito}')

    # --- Geração das retas (corrigidas) ---
    x_vals = np.linspace(X_original[:, 0].min() - 0.5, X_original[:, 0].max() + 0.5, 300)

    for digito in digitos_alvo[:-1]:
        if digito not in classificadores_scalers:
            continue

        clf, scaler = classificadores_scalers[digito]

        # Obter pesos e intercepto do modelo
        w = getattr(clf, 'coef_', None)
        b = getattr(clf, 'intercept_', 0)

        if w is None:
            print(f"[AVISO] Modelo do dígito {digito} não possui coeficientes (w).")
            continue

        # Flatten se necessário
        w = np.array(w).flatten()
        b = float(b)

        # Gerar reta no espaço escalado
        X_scaled = scaler.transform(np.c_[x_vals, np.zeros_like(x_vals)])
        x_scaled = X_scaled[:, 0]
        y_scaled = -(w[0] * x_scaled + b) / w[1]

        # Reverter normalização da coordenada y
        # (aplica inverso do scaler)
        X_unscaled = scaler.inverse_transform(np.c_[x_scaled, y_scaled])
        y_vals = X_unscaled[:, 1]

        plt.plot(x_vals, y_vals,
                 color=cores_linhas.get(digito, 'black'),
                 linestyle=estilos_linhas.get(digito, '--'),
                 linewidth=2,
                 label=f'Fronteira {digito} vs resto')

    # --- Ajustes visuais ---
    x_min, x_max = X_original[:, 0].min(), X_original[:, 0].max()
    y_min, y_max = X_original[:, 1].min(), X_original[:, 1].max()

    plt.xlim(x_min - 0.3, x_max + 0.3)
    plt.ylim(y_min - 0.3, y_max + 0.3)
    plt.xlabel("Intensidade", fontsize=12)
    plt.ylabel("Simetria", fontsize=12)
    plt.title("Fronteiras de decisão - Um Contra Todos", fontsize=14, weight='bold')
    plt.grid(alpha=0.3, linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.show()

"""
def plot_um_contra_todos(classificadores_scalers, X_original, y_original, digitos_alvo=[0,1,4,5]):
    cores_pontos = {0:'purple', 1:'lightgreen', 4:'salmon', 5:'violet'}
    marcadores = {0:'o', 1:'s', 4:'^', 5:'D'}
    cores_linhas = {0:'purple', 1:'lightgreen', 4:'salmon', 5:'violet'}
    estilos_linhas = {0:'--', 1:'-.', 4:':', 5:'-'}

    plt.figure(figsize=(10,8))

    # --- Plot dos pontos ---
    for digito in np.unique(y_original):
        mask = y_original == digito
        plt.scatter(X_original[mask,0], X_original[mask,1],
                    c=cores_pontos.get(digito, 'gray'),
                    marker=marcadores.get(digito, 'o'),
                    edgecolor='k', alpha=0.6,
                    label=f'Dígito {digito}')

    # --- Geração das retas (apenas 3, não 4) ---
    x_vals = np.linspace(X_original[:,0].min()-0.5, X_original[:,0].max()+0.5, 300)
    for digito in digitos_alvo[:-1]:
        if digito not in classificadores_scalers:
            continue
        clf, scaler = classificadores_scalers[digito]

        # Gera reta da fronteira (com bias correto)
        y_vals = calcula_y_model(clf, x_vals)
        plt.plot(x_vals, y_vals,
                 color=cores_linhas.get(digito, 'black'),
                 linestyle=estilos_linhas.get(digito, '--'),
                 linewidth=2,
                 label=f'Fronteira {digito} vs resto')

    plt.xlabel("Intensidade")
    plt.ylabel("Simetria")
    plt.title("Fronteiras de decisão - Um Contra Todos")
    plt.grid(alpha=0.3, linestyle=':')
    plt.legend()
    plt.show()



# === Funções para o método Um Contra Todos ===
def treinar_um_contra_todos(classificador, X_train, y_train, X_test, y_test, digitos_alvo=[0,1,4,5]):
    X_train_rest = X_train.copy()
    y_train_rest = y_train.copy()

    classificadores = {}
    scalers = {}

    for digito in digitos_alvo:
        y_bin = np.where(y_train_rest == digito, 1, -1)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train_rest)

        clf = classificador()
        clf.execute(X_scaled, y_bin)

        classificadores[digito] = clf
        scalers[digito] = scaler

        pred_train = clf.predict(X_scaled)
        mask_rest = pred_train != 1
        X_train_rest = X_train_rest[mask_rest]
        y_train_rest = y_train_rest[mask_rest]

        if len(X_train_rest) == 0:
            print(f"Todas as amostras foram classificadas antes de treinar o dígito {digito}.")
            break

    previsoes = []
    for x in X_test:
        pred = None
        for digito in digitos_alvo:
            clf = classificadores[digito]
            scaler = scalers[digito]
            x_scaled = scaler.transform([x])
            if clf.predict(x_scaled)[0] == 1:
                pred = digito
                break
        if pred is None:
            pred = 5
        previsoes.append(pred)

    print(f"\nModelo: {classificador.__name__}")
    print(classification_report(y_test, previsoes, digits=2))

    return {digito: (classificadores[digito], scalers[digito]) for digito in classificadores}, previsoes


def plot_um_contra_todos(classificadores_scalers, X_original, y_original, digitos_alvo=[0,1,4,5]):
    cores_pontos = {0:'purple', 1:'lightgreen', 4:'salmon', 5:'violet'}
    marcadores = {0:'o', 1:'s', 4:'^', 5:'D'}
    cores_linhas = {0:'purple', 1:'lightgreen', 4:'salmon', 5:'violet'}
    estilos_linhas = {0:'--', 1:'-.', 4:':', 5:'-'}

    plt.figure(figsize=(10,8))

    for digito in np.unique(y_original):
        mask = y_original == digito
        plt.scatter(X_original[mask,0], X_original[mask,1],
                    c=cores_pontos.get(digito, 'gray'),
                    marker=marcadores.get(digito, 'o'),
                    edgecolor='k', alpha=0.6,
                    label=f'Dígito {digito}')

    x_min, x_max = X_original[:,0].min() - 0.5, X_original[:,0].max() + 0.5
    y_min, y_max = X_original[:,1].min() - 0.5, X_original[:,1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 350),
                         np.linspace(y_min, y_max, 350))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    for digito in digitos_alvo:
        if digito not in classificadores_scalers:
            continue
        clf, scaler = classificadores_scalers[digito]
        grid_scaled = scaler.transform(grid_points)
        Z = clf.predict(grid_scaled)
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, levels=[0],
                    colors=cores_linhas.get(digito, 'black'),
                    linestyles=estilos_linhas.get(digito, '--'),
                    linewidths=2,
                    label=f'Fronteira {digito}')

    plt.xlabel("Intensidade")
    plt.ylabel("Simetria")
    plt.title("Fronteiras de decisão - Um Contra Todos")
    plt.grid(alpha=0.3, linestyle=':')
    plt.legend()
    plt.show()"""