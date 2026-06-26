"""
Análise das Métricas do Piano com Visão Computacional
=======================================================
Lê 'metricas_piano.csv' (gerado por piano.py) e produz gráficos +
um resumo estatístico, prontos para colar no relatório.

Uso:
    python3 analisar_metricas.py
    python3 analisar_metricas.py caminho/outro_arquivo.csv

Saída:
    graficos_metricas/  (pasta com os .png)
    resumo_metricas.txt
"""

import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ARQUIVO_CSV_PADRAO = "metricas_piano.csv"
PASTA_SAIDA = "graficos_metricas"


def carregar_dados(caminho):
    if not os.path.exists(caminho):
        print(f"ERRO: arquivo '{caminho}' não encontrado.")
        print("Execute piano.py primeiro para gerar o log de métricas.")
        sys.exit(1)

    df = pd.read_csv(caminho)

    # Colunas numéricas vêm como string quando vazias (latência sem nota disparada)
    colunas_numericas = [
        "tempo_inferencia_ms", "tempo_frame_ms", "fps_instantaneo",
        "brilho_medio", "desvio_brilho", "latencia_gesto_som_ms",
    ]
    for col in colunas_numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def resumo_estatistico(df):
    """Gera texto com estatísticas descritivas — direto para o relatório."""
    linhas = []
    linhas.append("RESUMO ESTATÍSTICO — SESSÃO DE CAPTURA")
    linhas.append("=" * 45)
    linhas.append(f"Total de frames processados: {len(df)}")
    linhas.append(f"Duração aproximada da sessão: {len(df) / df['fps_instantaneo'].mean():.1f} s "
                   f"(estimada por FPS médio)")
    linhas.append("")

    linhas.append("-- Desempenho do pipeline --")
    linhas.append(f"FPS médio:                {df['fps_instantaneo'].mean():.2f}")
    linhas.append(f"FPS mínimo / máximo:      {df['fps_instantaneo'].min():.2f} / {df['fps_instantaneo'].max():.2f}")
    linhas.append(f"Tempo de frame médio:     {df['tempo_frame_ms'].mean():.2f} ms")
    linhas.append("")

    linhas.append("-- Inferência do modelo (MediaPipe) --")
    linhas.append(f"Tempo médio de inferência: {df['tempo_inferencia_ms'].mean():.2f} ms")
    linhas.append(f"Desvio padrão:             {df['tempo_inferencia_ms'].std():.2f} ms")
    linhas.append(f"Percentil 95 (pior caso):  {df['tempo_inferencia_ms'].quantile(0.95):.2f} ms")
    linhas.append("")

    lat = df["latencia_gesto_som_ms"].dropna()
    linhas.append("-- Latência gesto → som --")
    if len(lat) > 0:
        linhas.append(f"Notas disparadas na sessão: {len(lat)}")
        linhas.append(f"Latência média:             {lat.mean():.2f} ms")
        linhas.append(f"Latência mínima / máxima:   {lat.min():.2f} / {lat.max():.2f} ms")
        linhas.append(f"Desvio padrão:              {lat.std():.2f} ms")
    else:
        linhas.append("Nenhuma nota foi disparada nesta sessão.")
    linhas.append("")

    linhas.append("-- Detecção de mãos / iluminação --")
    taxa_deteccao = 100.0 * (df["mao_detectada"] == 1).mean()
    linhas.append(f"Taxa de detecção de mão:   {taxa_deteccao:.1f}%")
    linhas.append(f"Brilho médio do ambiente:  {df['brilho_medio'].mean():.1f} / 255")
    linhas.append(f"Desvio padrão do brilho:   {df['brilho_medio'].std():.1f}")

    # Correlação simples entre brilho e detecção — relevante pro relatório
    corr = df["brilho_medio"].corr(df["mao_detectada"])
    linhas.append(f"Correlação brilho vs. detecção de mão: {corr:.3f}")

    texto = "\n".join(linhas)
    print(texto)

    with open("resumo_metricas.txt", "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    print(f"\nResumo salvo em 'resumo_metricas.txt'.")


def grafico_fps_ao_longo_do_tempo(df, pasta):
    plt.figure(figsize=(10, 4))
    plt.plot(df["frame"], df["fps_instantaneo"], linewidth=0.8, color="#2a7de1")
    media = df["fps_instantaneo"].mean()
    plt.axhline(media, color="red", linestyle="--", linewidth=1, label=f"Média: {media:.1f} FPS")
    plt.xlabel("Frame")
    plt.ylabel("FPS instantâneo")
    plt.title("FPS ao longo da sessão")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pasta, "fps_ao_longo_do_tempo.png"), dpi=150)
    plt.close()


def grafico_tempo_inferencia(df, pasta):
    plt.figure(figsize=(10, 4))
    plt.hist(df["tempo_inferencia_ms"].dropna(), bins=40, color="#e1812a", edgecolor="black")
    plt.xlabel("Tempo de inferência (ms)")
    plt.ylabel("Número de frames")
    plt.title("Distribuição do tempo de inferência do modelo (MediaPipe)")
    plt.tight_layout()
    plt.savefig(os.path.join(pasta, "histograma_tempo_inferencia.png"), dpi=150)
    plt.close()


def grafico_latencia_gesto_som(df, pasta):
    lat = df["latencia_gesto_som_ms"].dropna()
    if len(lat) == 0:
        print("Aviso: sem notas disparadas, gráfico de latência não gerado.")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(lat.values, marker="o", markersize=3, linewidth=0.8, color="#2aa84a")
    plt.axhline(lat.mean(), color="red", linestyle="--", linewidth=1,
                label=f"Média: {lat.mean():.1f} ms")
    plt.xlabel("Nota disparada (ordem cronológica)")
    plt.ylabel("Latência gesto → som (ms)")
    plt.title("Latência de resposta do instrumento")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pasta, "latencia_gesto_som.png"), dpi=150)
    plt.close()


def grafico_brilho_vs_deteccao(df, pasta):
    """
    Agrupa por faixas de brilho e mostra a taxa de detecção de mão em
    cada faixa — gráfico central para discutir 'qualidade do modelo em
    diferentes ambientes de luz' no relatório.
    """
    bins = np.arange(0, 261, 20)
    df = df.copy()
    df["faixa_brilho"] = pd.cut(df["brilho_medio"], bins=bins)

    taxa_por_faixa = df.groupby("faixa_brilho")["mao_detectada"].mean() * 100
    taxa_por_faixa = taxa_por_faixa.dropna()

    plt.figure(figsize=(10, 4))
    rotulos = [f"{int(i.left)}-{int(i.right)}" for i in taxa_por_faixa.index]
    plt.bar(rotulos, taxa_por_faixa.values, color="#9b59b6", edgecolor="black")
    plt.xlabel("Faixa de brilho médio (0-255)")
    plt.ylabel("Taxa de detecção de mão (%)")
    plt.title("Qualidade de detecção por condição de iluminação")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(pasta, "brilho_vs_deteccao.png"), dpi=150)
    plt.close()


def grafico_dispersao_brilho_inferencia(df, pasta):
    """Dispersão simples: brilho vs. tempo de inferência (custo do modelo muda com a cena?)."""
    plt.figure(figsize=(10, 4))
    plt.scatter(df["brilho_medio"], df["tempo_inferencia_ms"], s=6, alpha=0.4, color="#1abc9c")
    plt.xlabel("Brilho médio (0-255)")
    plt.ylabel("Tempo de inferência (ms)")
    plt.title("Brilho do ambiente vs. tempo de inferência do modelo")
    plt.tight_layout()
    plt.savefig(os.path.join(pasta, "dispersao_brilho_inferencia.png"), dpi=150)
    plt.close()


def main():
    caminho_csv = sys.argv[1] if len(sys.argv) > 1 else ARQUIVO_CSV_PADRAO

    print(f"Lendo '{caminho_csv}'...\n")
    df = carregar_dados(caminho_csv)

    os.makedirs(PASTA_SAIDA, exist_ok=True)

    resumo_estatistico(df)

    print("\nGerando gráficos...")
    grafico_fps_ao_longo_do_tempo(df, PASTA_SAIDA)
    grafico_tempo_inferencia(df, PASTA_SAIDA)
    grafico_latencia_gesto_som(df, PASTA_SAIDA)
    grafico_brilho_vs_deteccao(df, PASTA_SAIDA)
    grafico_dispersao_brilho_inferencia(df, PASTA_SAIDA)

    print(f"\nPronto! Gráficos salvos em '{PASTA_SAIDA}/':")
    for nome in sorted(os.listdir(PASTA_SAIDA)):
        print(f"  - {nome}")


if __name__ == "__main__":
    main()