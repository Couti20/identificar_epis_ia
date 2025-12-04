import cv2
import time
from threading import Thread
from ultralytics import YOLO

# --- CONFIGURAÇÕES EQUILIBRADAS PARA CELERON ---
ARQUIVO_MODELO = 'best.pt'
CONFIDENCE = 0.35           # 35% (Equilíbrio para pegar óculos sem pegar lixo)
TAMANHO_IMG_YOLO = 352      # O SEGREDO: 352 é maior que 320 (vê melhor) e muito mais leve que 640.
CAMERA_INDEX = 1            # 1 = USB (Sua câmera boa), 0 = Notebook
PULAR_QUADROS = 3           # Analisa 1 quadro e descansa 3. Isso libera o processador.

# --- NOMES REAIS ---
NOMES_CLASSES = {
    0: 'Pessoa', 1: 'Orelha', 2: 'Protetor Auricular', 3: 'Rosto', 
    4: 'Protetor Facial', 5: 'Mascara', 6: 'Pe', 7: 'Ferramenta', 
    8: 'Oculos', 9: 'Luvas', 10: 'Capacete', 11: 'Maos', 
    12: 'Cabeca', 13: 'Traje Medico', 14: 'Botas', 
    15: 'Traje Seguranca', 16: 'Colete'
}

# --- FILTRO DO QUE APARECE ---
ITENS_ATIVOS = [
    10, 16, 8, 9, 14,   # Seguros
    12, 3, 11, 6        # Perigos
]

VERDE = (0, 255, 0)
VERMELHO = (0, 0, 255)
AMARELO = (0, 255, 255)
PRETO = (0, 0, 0)
BRANCO = (255, 255, 255)

# --- CLASSE DE ACELERAÇÃO (Webcam em HD 720p) ---
# Usamos 720p em vez de 1080p para aliviar a carga de desenho na tela
class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.stream.isOpened(): self.stopped = True
        else:
            (self.grabbed, self.frame) = self.stream.read()
            self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped: return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self): return self.frame
    def stop(self): self.stopped = True

# --- LÓGICA DE SOBREPOSIÇÃO ---
def tem_sobreposicao(box_base, box_epi):
    x1, y1, x2, y2 = map(int, box_base.xyxy[0])
    e_x1, e_y1, e_x2, e_y2 = map(int, box_epi.xyxy[0])

    inter_x1 = max(x1, e_x1)
    inter_y1 = max(y1, e_y1)
    inter_x2 = min(x2, e_x2)
    inter_y2 = min(y2, e_y2)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2: return True 
    return False

# --- PROCESSAMENTO INTELIGENTE ---
def processar_frame(frame, results):
    boxes = results[0].boxes
    pessoas, coletes, capacetes, cabecas = [], [], [], []
    rostos, oculos, maos, luvas, pes, botas = [], [], [], [], [], []

    for box in boxes:
        cls = int(box.cls[0])
        if cls not in ITENS_ATIVOS and cls != 0: continue 

        if cls == 0: pessoas.append(box)
        elif cls == 16: coletes.append(box)
        elif cls == 10: capacetes.append(box)
        elif cls == 12: cabecas.append(box)
        elif cls == 3: rostos.append(box)
        elif cls == 8: oculos.append(box)
        elif cls == 11: maos.append(box)
        elif cls == 9: luvas.append(box)
        elif cls == 6: pes.append(box)
        elif cls == 14: botas.append(box)

    boxes_finais = [] 

    # 1. COLETE
    for pessoa in pessoas:
        tem = False
        for epi in coletes:
            if tem_sobreposicao(pessoa, epi): tem = True; break
        if not tem: boxes_finais.append({'box': pessoa, 'cor': VERMELHO, 'msg': "SEM COLETE"})
    for x in coletes: boxes_finais.append({'box': x, 'cor': VERDE, 'msg': "OK: COLETE"})

    # 2. CAPACETE
    for cabeca in cabecas:
        tem = False
        for epi in capacetes:
            if tem_sobreposicao(cabeca, epi): tem = True; break
        if not tem: boxes_finais.append({'box': cabeca, 'cor': VERMELHO, 'msg': "SEM CAPACETE"})
    for x in capacetes: boxes_finais.append({'box': x, 'cor': VERDE, 'msg': "OK: CAPACETE"})

    # 3. ÓCULOS
    for rosto in rostos:
        tem = False
        for epi in oculos:
            if tem_sobreposicao(rosto, epi): tem = True; break
        if not tem: boxes_finais.append({'box': rosto, 'cor': VERMELHO, 'msg': "SEM OCULOS"})
    for x in oculos: boxes_finais.append({'box': x, 'cor': VERDE, 'msg': "OK: OCULOS"})

    # 4. LUVAS
    for mao in maos:
        tem = False
        for epi in luvas:
            if tem_sobreposicao(mao, epi): tem = True; break
        if not tem: boxes_finais.append({'box': mao, 'cor': VERMELHO, 'msg': "SEM LUVA"})
    for x in luvas: boxes_finais.append({'box': x, 'cor': VERDE, 'msg': "OK: LUVA"})

    # 5. BOTAS
    for pe in pes:
        tem = False
        for epi in botas:
            if tem_sobreposicao(pe, epi): tem = True; break
        if not tem: boxes_finais.append({'box': pe, 'cor': VERMELHO, 'msg': "SEM BOTA"})
    for x in botas: boxes_finais.append({'box': x, 'cor': VERDE, 'msg': "OK: BOTA"})

    # DESENHO FINAL
    for item in boxes_finais:
        box = item['box']
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), item['cor'], 2)
        
        # Texto com fundo
        (w, h), _ = cv2.getTextSize(item['msg'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), item['cor'], -1)
        cv2.putText(frame, item['msg'], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    BRANCO if item['cor'] == VERMELHO else PRETO, 1)

    # Painel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, 80), PRETO, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, f"PESSOAS: {len(pessoas)}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, AMARELO, 1)
    
    status_oculos = "DETECTADO" if len(oculos) > 0 else "---"
    cor_txt = VERDE if len(oculos) > 0 else BRANCO
    cv2.putText(frame, f"OCULOS: {status_oculos}", (10, 65), cv2.FONT_HERSHEY_DUPLEX, 0.6, cor_txt, 1)

    return frame

# --- PROGRAMA ---
print("🚀 Iniciando Sistema Otimizado (352px)...")
model = YOLO(ARQUIVO_MODELO)

print(f"📷 Conectando Câmera {CAMERA_INDEX}...")
webcam = WebcamStream(src=CAMERA_INDEX).start()
time.sleep(1.5)

if webcam.stopped:
    print("❌ Erro na câmera. Mude CAMERA_INDEX para 0 ou 2.")
    exit()

# Tela Cheia
NOME_JANELA = "Monitoramento EPI - Otimizado"
cv2.namedWindow(NOME_JANELA, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(NOME_JANELA, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("✅ RODANDO! (Q para sair)")

frame_count = 0
ultimo_frame = None

while True:
    frame = webcam.read()
    if frame is None: continue

    frame_count += 1

    # Pula quadros para manter o vídeo rápido
    if frame_count % PULAR_QUADROS == 0:
        # 352px = Melhor compromisso entre velocidade e detalhe
        results = model(frame, imgsz=TAMANHO_IMG_YOLO, conf=CONFIDENCE, verbose=False)
        ultimo_frame = processar_frame(frame.copy(), results)
    
    # Exibe o último processamento ou o frame atual se ainda não processou
    imagem_final = ultimo_frame if ultimo_frame is not None else frame
    
    # Adiciona FPS
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - frame_count) # Estimativa simples
    # cv2.putText(imagem_final, "Otimizado", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BRANCO, 1)

    cv2.imshow(NOME_JANELA, imagem_final)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

webcam.stop()
cv2.destroyAllWindows()