import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler


# === Função auxiliar para plotar pontos corretamente e incorretamente classificados ===
def plot_points(X, y_true, y_preds, label_prefix=""):
    colors = ['blue', 'red']
    markers = ['o', 'x']

    y_pred = y_preds[0]

    plt.scatter(X[(y_true==1)&(y_pred==1),0], X[(y_true==1)&(y_pred==1),1], color=colors[0], label=f'{label_prefix} 1 correto', edgecolor='k', s=60)
    plt.scatter(X[(y_true==-1)&(y_pred==-1),0], X[(y_true==-1)&(y_pred==-1),1], color=colors[1], label=f'{label_prefix} 5 correto', edgecolor='k', s=60)

    plt.scatter(X[(y_true==1)&(y_pred==-1),0], X[(y_true==1)&(y_pred==-1),1], color=colors[0], marker='x', s=80, label=f'{label_prefix} 1 incorreto')
    plt.scatter(X[(y_true==-1)&(y_pred==1),0], X[(y_true==-1)&(y_pred==1),1],color=colors[1], marker='x', s=80, label=f'{label_prefix} 5 incorreto')


# === Função para calcular y do modelo para plotar a fronteira de decisão ===
def calcula_y_model(model, x_vals, titulo):
    if titulo.lower().startswith("regressão linear"):

        return -(model.w[0]*x_vals + model.w[2]) / model.w[1]
    else:

        return -(model.w[0] + model.w[1]*x_vals) / model.w[2]


# === Função para plotar resultados de um modelo específico com dados normalizados ===
def plot_model_results_normalized(model, X_train, y_train, X_test, y_test, scaler, titulo="Modelo"):
    """
    Mostra a fronteira de decisão e o desempenho do modelo nos conjuntos de treino e teste.
    Funciona com PocketPLA, Regressão Linear e Regressão Logística.
    """
    # Normaliza os conjuntos de treino e teste
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Gera uma faixa de valores para desenhar a reta
    x_vals = np.linspace(-0.25, 1.25, 300)

    #  🔹 GRÁFICO - TREINO
    plt.figure(figsize=(8,6))
    plt.scatter(X_train_scaled[y_train==1,0],  X_train_scaled[y_train==1,1],  label='Treino: 1 (+1)',  alpha=0.7)
    plt.scatter(X_train_scaled[y_train==-1,0], X_train_scaled[y_train==-1,1], label='Treino: 5 (-1)', alpha=0.7)

    # ======= Fronteira de decisão (para todos os modelos) =======
    w = model.w
    if len(w) == 3:  # inclui bias
        if not np.isclose(w[2], 0.0):
            y_vals = (-w[0] - w[1]*x_vals) / w[2]
            plt.plot(x_vals, y_vals, 'k-', linewidth=2, label=f'Fronteira {titulo}')
        else:
            plt.axvline(-w[0]/w[1], linewidth=2, color='k', label=f'Fronteira {titulo}')
    else:
        # fallback para caso raro sem bias separado
        if not np.isclose(w[1], 0.0):
            y_vals = (-w[0]) / w[1]
            plt.axvline(y_vals, linewidth=2, color='k', label=f'Fronteira {titulo}')

    plt.xlabel("Intensidade (normalizada)")
    plt.ylabel("Simetria (normalizada)")
    plt.title(f"{titulo} - Treino (normalizado)")
    plt.legend()
    plt.grid(True)
    plt.xlim(-0.25, 1.25)
    plt.ylim(-0.25, 1.25)
    plt.show()

    #  🔹 GRÁFICO - TESTE
    plt.figure(figsize=(8,6))

    # Predições (diferente para regressão linear)
    if titulo.lower().startswith("regressão linear"):
        yt = model.predict(X_test_scaled, class_output=True)
    else:
        yt = model.predict(X_test_scaled)

    # Pontos corretamente classificados
    plt.scatter(X_test_scaled[(y_test==1)&(yt==1),0],   X_test_scaled[(y_test==1)&(yt==1),1],   label='Teste: 1 correto', alpha=0.8)
    plt.scatter(X_test_scaled[(y_test==-1)&(yt==-1),0], X_test_scaled[(y_test==-1)&(yt==-1),1], label='Teste: 5 correto', alpha=0.8)
    
    # Pontos incorretamente classificados
    plt.scatter(X_test_scaled[(y_test==1)&(yt==-1),0],  X_test_scaled[(y_test==1)&(yt==-1),1],  marker='x', s=80, label='Teste: 1 incorreto')
    plt.scatter(X_test_scaled[(y_test==-1)&(yt==1),0],  X_test_scaled[(y_test==-1)&(yt==1),1],  marker='x', s=80, label='Teste: 5 incorreto')

    # Fronteira de decisão novamente
    if len(w) == 3 and not np.isclose(w[2], 0.0):
        y_vals = (-w[0] - w[1]*x_vals) / w[2]
        plt.plot(x_vals, y_vals, 'k-', linewidth=2, label=f'Fronteira {titulo}')
    elif len(w) == 3:
        plt.axvline(-w[0]/w[1], linewidth=2, color='k', label=f'Fronteira {titulo}')

    plt.xlabel("Intensidade (normalizada)")
    plt.ylabel("Simetria (normalizada)")
    plt.title(f"{titulo} - Teste (normalizado)")
    plt.legend()
    plt.grid(True)
    plt.xlim(-0.25, 1.25)
    plt.ylim(-0.25, 1.25)
    plt.show()


# === Função para comparar múltiplos modelos ===
def plot_all_models_comparison(X_train, y_train, X_test, y_test, models, model_names, scale='original', title="Comparação de Modelos Lineares"):
    # Escalonamento se necessário
    X_all = np.vstack([X_train, X_test])
    if scale == 'standard':
        scaler = StandardScaler()
        X_train_plot = scaler.fit_transform(X_train)
        X_test_plot  = scaler.transform(X_test)
        X_all_plot   = np.vstack([X_train_plot, X_test_plot])
    else:
        X_train_plot, X_test_plot, X_all_plot = X_train, X_test, X_all

    # Malha de pontos para o plano
    x_min, x_max = X_all_plot[:, 0].min() - 0.1, X_all_plot[:, 0].max() + 0.1
    y_min, y_max = X_all_plot[:, 1].min() - 0.1, X_all_plot[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Cálculo das fronteiras
    Zs = []
    for model in models:
        Z = model.predict(grid)
        # Ajuste para Regressão Linear → converte para classes
        if not np.isin(Z, [-1, 1]).all():
            Z = np.where(Z >= 0, 1, -1)
        Zs.append(Z.reshape(xx.shape))

    colors_models = ['green', 'purple', 'orange']

    # Previsões
    y_train_pred = [m.predict(X_train_plot) for m in models]
    y_test_pred  = [m.predict(X_test_plot)  for m in models]
    # Converte previsões contínuas (caso LinearRegression)
    for i in range(len(y_train_pred)):
        if not np.isin(y_train_pred[i], [-1, 1]).all():
            y_train_pred[i] = np.where(y_train_pred[i] >= 0, 1, -1)
            y_test_pred[i]  = np.where(y_test_pred[i]  >= 0, 1, -1)

    # === GRÁFICO TREINAMENTO ===
    plt.figure(figsize=(10, 7))
    handles = []
    for i, Z in enumerate(Zs):
        plt.contour(xx, yy, Z, colors=colors_models[i], linewidths=2, levels=[0], linestyles='--')
        handles.append(mpatches.Patch(color=colors_models[i], label=f'Fronteira {model_names[i]}'))
    plot_points(X_train_plot, y_train, y_train_pred[0], label_prefix="Treino")
    plt.xlabel("Intensidade")
    plt.ylabel("Simetria")
    plt.title(title + " - Treino")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(handles=handles, fontsize=9, loc='upper right')
    plt.show()

    # === GRÁFICO TESTE ===
    plt.figure(figsize=(10, 7))
    handles = []
    for i, Z in enumerate(Zs):
        plt.contour(xx, yy, Z, colors=colors_models[i], linewidths=2, levels=[0], linestyles='--')
        handles.append(mpatches.Patch(color=colors_models[i], label=f'Fronteira {model_names[i]}'))
    plot_points(X_test_plot, y_test, y_test_pred[0], label_prefix="Teste")
    plt.xlabel("Intensidade")
    plt.ylabel("Simetria")
    plt.title(title + " - Teste")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(handles=handles, fontsize=9, loc='upper right')
    plt.show()