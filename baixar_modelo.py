import urllib.request

def baixar_com_progresso(url, caminho):
    def reporthook(count, block_size, total_size):
        baixado = count * block_size
        pct = baixado / total_size * 100 if total_size > 0 else 0
        mb = baixado / 1024 / 1024
        print(f"\r  {pct:.1f}% — {mb:.1f} MB baixados", end="", flush=True)

    print("Baixando hand_landmarker.task...")
    urllib.request.urlretrieve(url, caminho, reporthook)
    print("\nConcluído!")

baixar_com_progresso(
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "hand_landmarker.task"
)