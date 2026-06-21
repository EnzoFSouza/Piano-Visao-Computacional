import pygame
import sys

# Inicializa o Pygame e o mixer de áudio
pygame.init()
pygame.mixer.init()

# Configurações da janela
LARGURA_TELA = 600
ALTURA_TELA = 400
tela = pygame.display.set_mode((LARGURA_TELA, ALTURA_TELA))
pygame.display.set_caption("Meu Piano em Python")

# Cores
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
CINZA = (200, 200, 200)

DIRETORIO = "notas_piano"

# 1. Mapeamento das teclas do PC -> Arquivos de áudio
notas_arquivos = {
    pygame.K_1: f"{DIRETORIO}/E3.wav",
    pygame.K_2: f"{DIRETORIO}/A3.wav",
    pygame.K_3: f"{DIRETORIO}/B3.wav",

    pygame.K_4: f"{DIRETORIO}/C4.wav",
    pygame.K_5: f"{DIRETORIO}/D4.wav",
    pygame.K_6: f"{DIRETORIO}/E4.wav",
}

# 2. Carregar os sons no Pygame
# Criar dicionário onde o Pygame guarda o áudio carregado na memória
piano_sons = {}
for tecla, arquivo in notas_arquivos.items():
    try:
        piano_sons[tecla] = pygame.mixer.Sound(arquivo)
    except FileNotFoundError:
        # Se não tiver os arquivos ainda, o código roda sem som para testar a interface
        piano_sons[tecla] = None
        print(f"Aviso: Arquivo {arquivo} não encontrado. O piano funcionará apenas visualmente.")

# Dicionário para rastrear quais teclas estão sendo pressionadas no momento (para mudar a cor)
teclas_pressionadas = {tecla: False for tecla in notas_arquivos}

# Configurações de desenho das 10 teclas virtuais
qtd_teclas = 6
largura_tecla = LARGURA_TELA // qtd_teclas

# Loop principal
while True:
    tela.fill(PRETO)

    # Captura de eventos (cliques, teclas, fechar janela)
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # Quando uma tecla é pressionada
        if evento.type == pygame.KEYDOWN:
            if evento.key in piano_sons:
                teclas_pressionadas[evento.key] = True
                if piano_sons[evento.key]:
                    piano_sons[evento.key].play() # Toca o som da nota

        # Quando a tecla é solta
        if evento.type == pygame.KEYUP:
            if evento.key in piano_sons:
                teclas_pressionadas[evento.key] = False
        
    # 3. Desenhar as N teclas na tela
    for i, tecla_codigo in enumerate(notas_arquivos.keys()):
        # Define a cor: se estiver pressionada fica cinza, se não, fica branca
        cor = CINZA if teclas_pressionadas[tecla_codigo] else BRANCO
        
        # Calcula a posição X de cada tecla na tela
        pos_x = i * largura_tecla
        
        # Desenha o retângulo da tecla (deixando 2 pixels de borda preta entre elas)
        pygame.draw.rect(tela, cor, (pos_x, 50, largura_tecla - 2, ALTURA_TELA - 100))

    pygame.display.flip() # Atualiza a tela