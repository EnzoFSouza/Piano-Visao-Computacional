import numpy as np
import wave
import os

# Configurações do áudio
TAXA_AMOSTRAGEM = 44100
DURACAO = 2.5  # Notas sustentadas, boas para piano

DIRETORIO = "notas_piano"

NOMES_NOTAS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def gerar_frequencias_piano():
    """
    Gera todas as 88 notas de um piano padrão, de A0 até C8,
    usando afinação igual (A4 = 440 Hz) como referência.

    Fórmula: f = 440 * 2^((n - 49) / 12)
    onde n é a posição da tecla no piano (A0 = 1, C8 = 88).
    """
    notas = {}

    # A0 é a primeira tecla do piano (n = 1)
    # Iterar por oitava e nota
    midi_a0 = 21  # número MIDI da nota A0

    for midi in range(midi_a0, midi_a0 + 88):
        nota_index = (midi - 12) % 12
        oitava = (midi - 12) // 12
        nome_nota = NOMES_NOTAS[nota_index]
        frequencia = 440.0 * (2 ** ((midi - 69) / 12))

        nome_arquivo = f"{nome_nota}{oitava}.wav"
        notas[nome_arquivo] = round(frequencia, 2)

    return notas


def gerar_onda_som(frequencia, duracao, taxa):
    """
    Gera timbre de piano com:
    - Envelope ADSR: ataque ~15ms, decay/release exponencial
    - Harmônicos (fundamentais + 5 parciais) para corpo sonoro rico
    - Leve vibrato para suavizar notas longas
    """
    t = np.linspace(0, duracao, int(taxa * duracao), False)

    # Onda com harmônicos
    onda  = 1.00 * np.sin(2 * np.pi * frequencia * t)
    onda += 0.50 * np.sin(2 * np.pi * (2 * frequencia) * t)
    onda += 0.25 * np.sin(2 * np.pi * (3 * frequencia) * t)
    onda += 0.12 * np.sin(2 * np.pi * (4 * frequencia) * t)
    onda += 0.06 * np.sin(2 * np.pi * (5 * frequencia) * t)
    onda += 0.03 * np.sin(2 * np.pi * (6 * frequencia) * t)

    # Leve vibrato (profundidade 1 Hz, taxa 5 Hz) — dá "vida" às notas sustentadas
    vibrato = 1 + 0.003 * np.sin(2 * np.pi * 5 * t)
    onda = onda * vibrato

    # Envelope: ataque em 15ms, depois decay exponencial suave
    ataque_amostras = int(0.015 * taxa)
    envelope = np.exp(-2.0 * t)
    envelope[:ataque_amostras] = np.linspace(0, 1, ataque_amostras)
    onda = onda * envelope

    # Normaliza para 16-bits
    onda = onda / np.max(np.abs(onda))
    return (onda * 32767).astype(np.int16)


def salvar_wav(caminho, dados, taxa):
    with wave.open(caminho, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(taxa)
        f.writeframes(dados.tobytes())


if __name__ == "__main__":
    os.makedirs(DIRETORIO, exist_ok=True)
    notas_piano = gerar_frequencias_piano()

    print(f"Gerando {len(notas_piano)} notas em '{DIRETORIO}/'...\n")

    for nome, freq in notas_piano.items():
        caminho = os.path.join(DIRETORIO, nome)
        dados = gerar_onda_som(freq, DURACAO, TAXA_AMOSTRAGEM)
        salvar_wav(caminho, dados, TAXA_AMOSTRAGEM)
        print(f"  ✓ {nome:10s}  ({freq:.2f} Hz)")

    print(f"{len(notas_piano)} arquivos .wav gerados em '{DIRETORIO}/'.")