import pygame
import sys

# ==========================================================
# CONFIGURAÇÃO
# ==========================================================

pygame.init()
pygame.mixer.init()

LARGURA_TELA = 1400
ALTURA_TELA = 450

tela = pygame.display.set_mode((LARGURA_TELA, ALTURA_TELA))
pygame.display.set_caption("Piano Virtual")

BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
CINZA = (180, 180, 180)
AZUL = (120, 180, 255)

DIRETORIO = "notas_piano"

fonte_nota = pygame.font.SysFont(None, 24)
fonte_tecla = pygame.font.SysFont(None, 32)

# ==========================================================
# MAPEAMENTO DAS TECLAS
# ==========================================================

teclas = [
    pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5,
    pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9, pygame.K_0,

    pygame.K_q, pygame.K_w, pygame.K_e, pygame.K_r, pygame.K_t,
    pygame.K_y, pygame.K_u, pygame.K_i, pygame.K_o, pygame.K_p,

    pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_f,
    pygame.K_g, pygame.K_h, pygame.K_j, pygame.K_k, pygame.K_l
]

rotulos = [
    "1","2","3","4","5","6","7","8","9","0",
    "Q","W","E","R","T","Y","U","I","O","P",
    "A","S","D","F","G","H","J","K","L"
]

notas = [
    "C3","C#3","D3","D#3","E3",
    "F3","F#3","G3","G#3","A3",

    "A#3","B3","C4","C#4","D4",
    "D#4","E4","F4","F#4","G4",

    "G#4","A4","A#4","B4",
    "C5","C#5","D5","D#5","E5"
]

# ==========================================================
# CARREGAMENTO DOS SONS
# ==========================================================

piano_sons = {}

for tecla, nota in zip(teclas, notas):

    arquivo = f"{DIRETORIO}/{nota}.wav"

    try:
        piano_sons[tecla] = pygame.mixer.Sound(arquivo)

    except FileNotFoundError:

        piano_sons[tecla] = None
        print(f"Arquivo não encontrado: {arquivo}")

teclas_pressionadas = {
    tecla: False
    for tecla in teclas
}

# ==========================================================
# LAYOUT VISUAL
# ==========================================================

qtd_teclas = len(teclas)

largura_tecla = LARGURA_TELA // qtd_teclas

# ==========================================================
# LOOP PRINCIPAL
# ==========================================================

while True:

    for evento in pygame.event.get():

        if evento.type == pygame.QUIT:

            pygame.quit()
            sys.exit()

        elif evento.type == pygame.KEYDOWN:

            if evento.key in piano_sons:

                teclas_pressionadas[evento.key] = True

                if piano_sons[evento.key]:
                    piano_sons[evento.key].play()

        elif evento.type == pygame.KEYUP:

            if evento.key in piano_sons:

                teclas_pressionadas[evento.key] = False

    # ======================================================
    # DESENHO
    # ======================================================

    tela.fill(PRETO)

    for i, tecla in enumerate(teclas):

        x = i * largura_tecla

        nota_atual = notas[i]

        eh_sustenido = "#" in nota_atual

        if teclas_pressionadas[tecla]:

            if eh_sustenido:
                cor = (70, 130, 255)      # azul escuro
            else:
                cor = (120, 180, 255)     # azul claro

        else:
            cor = PRETO if eh_sustenido else BRANCO

        pygame.draw.rect(
            tela,
            cor,
            (
                x,
                40,
                largura_tecla - 2,
                ALTURA_TELA - 80
            )
        )

        pygame.draw.rect(
            tela,
            PRETO,
            (
                x,
                40,
                largura_tecla - 2,
                ALTURA_TELA - 80
            ),
            2
        )

        cor_texto = BRANCO if eh_sustenido else PRETO

        texto_nota = fonte_nota.render(
            notas[i],
            True,
            cor_texto
        )

        tela.blit(
            texto_nota,
            (
                x + largura_tecla//2 - texto_nota.get_width()//2,
                ALTURA_TELA - 120
            )
        )

        texto_tecla = fonte_tecla.render(
            rotulos[i],
            True,
            cor_texto
        )

        tela.blit(
            texto_tecla,
            (
                x + largura_tecla//2 - texto_tecla.get_width()//2,
                ALTURA_TELA - 80
            )
        )

    pygame.display.flip()