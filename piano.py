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

import sys
import time

import cv2
import mediapipe as mp
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
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            frame     = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp = int((frame_idx / fps_cam) * 1000)

            resultado           = landmarker.detect_for_video(mp_image, timestamp)
            notas_ativas_chaves = set()
            agora               = time.time()

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
            for chave in notas_ativas_chaves:
                if agora - ultimo_toque[chave] >= COOLDOWN_NOTA:
                    canais[chave].play(sons[chave])
                    ultimo_toque[chave] = agora
                    nome, _ = NOTAS_MAO[chave]
                    print(f"  ♪ {nome} ({chave[0]})")

            desenhar_hud(frame, notas_ativas_chaves)

            cv2.putText(frame, "Q = sair", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            cv2.imshow("Piano de Mao", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    print("Encerrado.")


if __name__ == "__main__":
    main()