"""
Piano com Visão Computacional
==============================
Integração: OpenCV (câmera) + MediaPipe (mãos) + Pygame (som)

Mapeamento — 3 dedos por mão, 6 notas no total:

  MÃO ESQUERDA          MÃO DIREITA
  Polegar  (4) -> E3    Polegar  (4) -> D4
  Indicador(8) -> A3    Indicador(8) -> E4  (landmark 8 compartilhado, nota depende da mão)
  Anelar  (16) -> B3    Anelar  (16) -> C4

Polegar: ativo quando a ponta cruza para dentro da palma (eixo X).
  Mão direita: ponta.x > base_palma.x
  Mão esquerda: ponta.x < base_palma.x
  (base_palma = landmark 9, raiz do dedo médio — centro estável da palma)

Outros dedos: ativos quando ponta.y > articulação_base.y (dedo dobrado).
"""

import csv
import os
import statistics
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
import pygame

# ──────────────────────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────────────────────

CAMINHO_MODELO = "hand_landmarker.task"
COOLDOWN_NOTA  = 0.50   # segundos mínimos entre ativações da mesma nota
DIRETORIO      = "notas_piano"

# ──────────────────────────────────────────────────────────────
# MÉTRICAS — CONFIGURAÇÃO
# ──────────────────────────────────────────────────────────────
# Todas as medições usam time.perf_counter() (relógio monotônico de
# alta resolução, recomendado para benchmarking — não sofre saltos
# por ajuste de hora do sistema, ao contrário de time.time()).

LOG_METRICAS       = True
ARQUIVO_LOG        = "metricas_piano.csv"
JANELA_MEDIA_MOVEL = 30   # nº de frames usados para médias/HUD em tempo real

# Notas por mão: landmark_id -> (nome, arquivo)
# Usamos chaves compostas (hand, lid) internamente
NOTAS_MAO = {
    # Mão esquerda  (MediaPipe label: "Left"  = mão esquerda do usuário com flip)
    #("Left",  4):  ("E3", f"{DIRETORIO}/E3.wav"),
    ("Left",  4):  ("A5", f"{DIRETORIO}/A5.wav"),
    ("Left",  8):  ("B5", f"{DIRETORIO}/B5.wav"),
    ("Left", 12):  ("C4", f"{DIRETORIO}/C4.wav"),
    # Mão direita
    ("Right",  4): ("E3", f"{DIRETORIO}/E3.wav"),
    ("Right",  8): ("B4", f"{DIRETORIO}/B4.wav"),
    ("Right", 12): ("F#3", f"{DIRETORIO}/F#3.wav"),
}

# Dedos monitorados (exceto polegar, tratado separado)
DEDOS_BASE = {8: 6, 12: 10}   # ponta -> articulação base

# Cores UI (BGR)
COR_NOTA_ATIVA   = (80, 220, 80)
COR_NOTA_INATIVA = (120, 120, 120)
COR_PONTA_ATIVA  = (60, 60, 255)
COR_LANDMARKS    = (0, 200, 0)
COR_CONEXOES     = (160, 160, 160)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# ──────────────────────────────────────────────────────────────
# COLETOR DE MÉTRICAS TÉCNICAS
# ──────────────────────────────────────────────────────────────

class ColetorMetricas:
    """
    Centraliza todas as medições de desempenho do pipeline:

      - tempo_inferencia_ms : tempo da chamada ao HandLandmarker
                              (gargalo típico do pipeline; isola o
                              custo do modelo do custo de I/O/desenho)
      - tempo_frame_ms      : tempo total do loop (captura + inferência
                              + lógica de gestos + desenho) -> FPS efetivo
      - latencia_gesto_som_ms : tempo entre o instante em que o gesto é
                              classificado como "ativo" num frame e o
                              instante em que pygame.Channel.play() é
                              chamado. É a métrica mais relevante para
                              UX: mede a responsividade real do "instrumento".
      - brilho_medio / desvio_brilho : média e desvio padrão da escala
                              de cinza do frame. Proxy quantificável de
                              "qualidade do ambiente de luz", para
                              correlacionar com taxa_deteccao depois.
      - mao_detectada       : booleano por frame -> taxa de detecção
                              acumulada (robustez do modelo na sessão).

    Os dados são acumulados em deques de tamanho fixo para exibir
    médias móveis no HUD, e cada frame é também persistido em CSV
    (se LOG_METRICAS=True) para análise posterior (ex. pandas/matplotlib
    no relatório: latência vs. brilho, FPS vs. nº de mãos na cena etc.).
    """

    def __init__(self, janela=JANELA_MEDIA_MOVEL, arquivo_log=ARQUIVO_LOG, ativar_log=LOG_METRICAS):
        self.janela = janela
        self.tempos_inferencia = []
        self.tempos_frame = []
        self.latencias_gesto_som = []
        self.frames_totais = 0
        self.frames_com_mao = 0

        self.ativar_log = ativar_log
        self._writer = None
        self._arquivo = None
        if self.ativar_log:
            novo = not os.path.exists(arquivo_log)
            self._arquivo = open(arquivo_log, "a", newline="")
            self._writer = csv.writer(self._arquivo)
            if novo:
                self._writer.writerow([
                    "frame", "timestamp_unix",
                    "tempo_inferencia_ms", "tempo_frame_ms", "fps_instantaneo",
                    "mao_detectada", "n_maos", "brilho_medio", "desvio_brilho",
                    "nota_disparada", "latencia_gesto_som_ms",
                ])

    # ---- registro por frame ----
    def registrar_frame(self, tempo_inferencia_s, tempo_frame_s,
                         mao_detectada, n_maos, brilho_medio, desvio_brilho,
                         nota_disparada="", latencia_gesto_som_s=None):
        self.frames_totais += 1
        if mao_detectada:
            self.frames_com_mao += 1

        t_inf_ms   = tempo_inferencia_s * 1000.0
        t_frame_ms = tempo_frame_s * 1000.0
        fps_inst   = (1.0 / tempo_frame_s) if tempo_frame_s > 0 else 0.0

        self.tempos_inferencia.append(t_inf_ms)
        self.tempos_frame.append(t_frame_ms)
        if len(self.tempos_inferencia) > self.janela:
            self.tempos_inferencia.pop(0)
            self.tempos_frame.pop(0)

        lat_ms = ""
        if latencia_gesto_som_s is not None:
            lat_ms = latencia_gesto_som_s * 1000.0
            self.latencias_gesto_som.append(lat_ms)
            if len(self.latencias_gesto_som) > self.janela:
                self.latencias_gesto_som.pop(0)

        if self._writer:
            self._writer.writerow([
                self.frames_totais, time.time(),
                f"{t_inf_ms:.3f}", f"{t_frame_ms:.3f}", f"{fps_inst:.2f}",
                int(mao_detectada), n_maos,
                f"{brilho_medio:.2f}", f"{desvio_brilho:.2f}",
                nota_disparada, f"{lat_ms:.3f}" if lat_ms != "" else "",
            ])

    # ---- agregados para HUD em tempo real ----
    def media_inferencia_ms(self):
        return statistics.mean(self.tempos_inferencia) if self.tempos_inferencia else 0.0

    def fps_medio(self):
        if not self.tempos_frame:
            return 0.0
        media_ms = statistics.mean(self.tempos_frame)
        return 1000.0 / media_ms if media_ms > 0 else 0.0

    def media_latencia_gesto_som_ms(self):
        return statistics.mean(self.latencias_gesto_som) if self.latencias_gesto_som else 0.0

    def taxa_deteccao_mao(self):
        if self.frames_totais == 0:
            return 0.0
        return 100.0 * self.frames_com_mao / self.frames_totais

    def fechar(self):
        if self._arquivo:
            self._arquivo.close()


def medir_brilho(frame_bgr):
    """
    Converte para escala de cinza e retorna (média, desvio padrão).
    Média alta = ambiente claro; desvio alto = boa separação mão/fundo
    (mais contraste). Útil para correlacionar qualidade de detecção
    com condições de iluminação em diferentes ambientes de teste.
    """
    cinza = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cinza)), float(np.std(cinza))


# ──────────────────────────────────────────────────────────────
# CARREGAMENTO DOS SONS
# ──────────────────────────────────────────────────────────────

def carregar_sons():
    sons = {}
    carregados = {}  # arquivo -> Sound (evita carregar o mesmo WAV duas vezes)
    for chave, (nome, arquivo) in NOTAS_MAO.items():
        if arquivo not in carregados:
            try:
                carregados[arquivo] = pygame.mixer.Sound(arquivo)
                print(f"  ✔ {nome} -> {arquivo}")
            except FileNotFoundError:
                print(f"  ✘ ERRO: arquivo não encontrado: {arquivo}")
                pygame.quit()
                sys.exit(1)
        sons[chave] = carregados[arquivo]
    return sons


# ──────────────────────────────────────────────────────────────
# LÓGICA DE DETECÇÃO
# ──────────────────────────────────────────────────────────────

def dedo_dobrado(landmarks, ponta_id, base_id):
    """Y cresce para baixo: ponta.y > base.y = dedo dobrado."""
    return landmarks[ponta_id].y > landmarks[base_id].y


def polegar_dentro_palma(landmarks, handedness):
    """
    Polegar ativo quando a ponta entra para dentro da palma.
    Usa landmark 9 (base do médio) como referência central da palma.
      Mão direita: ponta.x > palma.x  (ponta vai para direita = para dentro)
      Mão esquerda: ponta.x < palma.x (ponta vai para esquerda = para dentro)
    """
    ponta = landmarks[4]
    palma = landmarks[2] #2 é a base do polegar, e não 9 (base do médio, fica muito ruim com 9)
    if handedness == "Right":
        return ponta.x < palma.x
    else:
        return ponta.x > palma.x


# ──────────────────────────────────────────────────────────────
# DESENHO
# ──────────────────────────────────────────────────────────────

def desenhar_landmarks(frame, hand_lms):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], COR_CONEXOES, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 5, COR_LANDMARKS, -1)


def desenhar_ponta_ativa(frame, hand_lms, ponta_id):
    h, w = frame.shape[:2]
    cx = int(hand_lms[ponta_id].x * w)
    cy = int(hand_lms[ponta_id].y * h)
    cv2.circle(frame, (cx, cy), 14, COR_PONTA_ATIVA, -1)
    cv2.circle(frame, (cx, cy), 14, (255, 255, 255), 2)


def desenhar_hud(frame, notas_ativas_chaves):
    """
    Duas seções no HUD: mão esquerda (A3, B3, C4) e direita (E3, D4, E4).
    """
    h, w = frame.shape[:2]
    barra_h = 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - barra_h), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Ordem de exibição: esquerda → direita na tela
    layout = [
        ("Left",  4,  "Pol."),
        ("Left",  8,  "Ind."),
        ("Left", 12,  "Méd."),
        ("Right",  4, "Pol."),
        ("Right",  8, "Ind."),
        ("Right", 12, "Méd."),
    ]

    passo = w // len(layout)

    # Linha divisória entre mãos
    meio = w // 2
    cv2.line(frame, (meio, h - barra_h), (meio, h), (60, 60, 60), 2)
    cv2.putText(frame, "DIR", (meio - 120, h - barra_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)
    cv2.putText(frame, "ESQ", (meio + 20, h - barra_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)

    for i, (hand, lid, dedo_label) in enumerate(layout):
        chave = (hand, lid)
        nome, _ = NOTAS_MAO[chave]
        ativo = chave in notas_ativas_chaves
        cor   = COR_NOTA_ATIVA if ativo else COR_NOTA_INATIVA
        cx    = i * passo + passo // 2
        cy    = h - barra_h // 2

        raio = 22 if ativo else 18
        cv2.circle(frame, (cx, cy - 6), raio, cor, -1 if ativo else 2)
        cv2.putText(frame, nome, (cx - 16, cy - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0) if ativo else cor, 2 if ativo else 1)
        cv2.putText(frame, dedo_label, (cx - 16, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, cor, 1)


def desenhar_hud_metricas(frame, coletor, brilho_medio, n_maos):
    """
    HUD técnico no topo da tela: FPS médio, tempo de inferência do
    modelo, latência gesto->som e taxa de detecção de mãos na sessão.
    Pensado para aparecer em prints/vídeo do relatório.
    """
    linhas = [
        f"FPS: {coletor.fps_medio():.1f}",
        f"Inferencia: {coletor.media_inferencia_ms():.1f} ms",
        f"Latencia gesto->som: {coletor.media_latencia_gesto_som_ms():.1f} ms",
        f"Deteccao de mao (sessao): {coletor.taxa_deteccao_mao():.1f}%",
        f"Brilho medio: {brilho_medio:.0f}/255   Maos no frame: {n_maos}",
    ]

    x, y = 10, 55
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 40), (430, 40 + 20 * len(linhas) + 10), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for linha in linhas:
        cv2.putText(frame, linha, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 230, 255), 1, cv2.LINE_AA)
        y += 20


# ──────────────────────────────────────────────────────────────
# PROGRAMA PRINCIPAL
# ──────────────────────────────────────────────────────────────

def main():
    pygame.init()
    pygame.mixer.init(buffer=512)
    pygame.mixer.set_num_channels(16)

    print("Carregando sons...")
    sons = carregar_sons()
    print("Sons prontos.\n")

    # Canal dedicado por chave (hand, lid)
    canais = {chave: pygame.mixer.Channel(i) for i, chave in enumerate(NOTAS_MAO)}
    ultimo_toque = {chave: 0.0 for chave in NOTAS_MAO}

    coletor = ColetorMetricas()
    print(f"Métricas sendo registradas em '{ARQUIVO_LOG}'." if LOG_METRICAS else "Log de métricas desativado.")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=CAMINHO_MODELO),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERRO: câmera não encontrada.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps_cam = cap.get(cv2.CAP_PROP_FPS) or 30

    print("🎹 Piano iniciado!")
    print("   Mão esquerda:  Pol=E3  Ind=A3  Ane=B3")
    print("   Mão direita:   Pol=D4  Ind=E4  Ane=C4")
    print("   Pressione Q para sair.\n")

    frame_idx = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            t_inicio_frame = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            frame     = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp = int((frame_idx / fps_cam) * 1000)

            brilho_medio, desvio_brilho = medir_brilho(frame)

            t_antes_inferencia = time.perf_counter()
            resultado          = landmarker.detect_for_video(mp_image, timestamp)
            t_inferencia       = time.perf_counter() - t_antes_inferencia

            notas_ativas_chaves = set()
            agora               = time.time()
            t_gesto_detectado   = time.perf_counter()  # marca p/ latência gesto->som

            if resultado.hand_landmarks:
                for idx_mao, hand_lms in enumerate(resultado.hand_landmarks):
                    desenhar_landmarks(frame, hand_lms)

                    handedness = "Right"
                    if resultado.handedness and idx_mao < len(resultado.handedness):
                        handedness = resultado.handedness[idx_mao][0].category_name

                    # Polegar (lógica lateral — ponta dentro da palma)
                    if polegar_dentro_palma(hand_lms, handedness):
                        chave = (handedness, 4)
                        if chave in NOTAS_MAO:
                            notas_ativas_chaves.add(chave)
                            desenhar_ponta_ativa(frame, hand_lms, 4)

                    # Indicador e Anelar (lógica vertical)
                    for ponta_id, base_id in DEDOS_BASE.items():
                        if dedo_dobrado(hand_lms, ponta_id, base_id):
                            chave = (handedness, ponta_id)
                            if chave in NOTAS_MAO:
                                notas_ativas_chaves.add(chave)
                                desenhar_ponta_ativa(frame, hand_lms, ponta_id)

            # Tocar notas ativas com cooldown
            nota_disparada_neste_frame = ""
            latencia_gesto_som = None
            for chave in notas_ativas_chaves:
                if agora - ultimo_toque[chave] >= COOLDOWN_NOTA:
                    canais[chave].play(sons[chave])
                    latencia_gesto_som = time.perf_counter() - t_gesto_detectado
                    ultimo_toque[chave] = agora
                    nome, _ = NOTAS_MAO[chave]
                    nota_disparada_neste_frame = nome
                    print(f"  ♪ {nome} ({chave[0]})  [latência gesto→som: {latencia_gesto_som*1000:.1f} ms]")

            desenhar_hud(frame, notas_ativas_chaves)

            n_maos_no_frame = len(resultado.hand_landmarks) if resultado.hand_landmarks else 0
            tempo_frame_total = time.perf_counter() - t_inicio_frame

            coletor.registrar_frame(
                tempo_inferencia_s=t_inferencia,
                tempo_frame_s=tempo_frame_total,
                mao_detectada=n_maos_no_frame > 0,
                n_maos=n_maos_no_frame,
                brilho_medio=brilho_medio,
                desvio_brilho=desvio_brilho,
                nota_disparada=nota_disparada_neste_frame,
                latencia_gesto_som_s=latencia_gesto_som,
            )

            desenhar_hud_metricas(frame, coletor, brilho_medio, n_maos_no_frame)

            cv2.putText(frame, "Q = sair", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            cv2.imshow("Piano de Mao", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    coletor.fechar()

    print("\n── Resumo da sessão ──")
    print(f"  Frames processados:        {coletor.frames_totais}")
    print(f"  Taxa de detecção de mão:   {coletor.taxa_deteccao_mao():.1f}%")
    print(f"  FPS médio:                 {coletor.fps_medio():.1f}")
    print(f"  Inferência média (modelo): {coletor.media_inferencia_ms():.1f} ms")
    print(f"  Latência média gesto→som:  {coletor.media_latencia_gesto_som_ms():.1f} ms")
    if LOG_METRICAS:
        print(f"  Log completo salvo em:     {ARQUIVO_LOG}")
    print("Encerrado.")


if __name__ == "__main__":
    main()