import cv2
import time
from threading import Thread
from ultralytics import YOLO

# --- CONFIGURAÇÕES PARA NOTEBOOK CELERON/4GB ---
ARQUIVO_MODELO = 'best.pt'
CONFIDENCE = 0.30           # 30% de certeza já mostra (ajuda a pegar óculos)
TAMANHO_IMG_YOLO = 320      # 320x320 é o limite para Celeron. (640 travaria)
CAMERA_INDEX = 1            # 1 = USB, 0 = Notebook
PULAR_QUADROS = 4           # Analisa 1 frame, pula 3. Essencial para não travar.

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
    10, # Capacete
    16, # Colete
    8,  # Oculos
    9,  # Luvas
    14, # Botas
    12, # Cabeca (Perigo)
    3,  # Rosto (Perigo - Atenção aqui)
    11, # Maos (Perigo)
    6,  # Pe (Perigo)
]

VERDE = (0, 255, 0)
VERMELHO = (0, 0, 255)
AMARELO = (0, 255, 255)
PRETO = (0, 0, 0)
BRANCO = (255, 255, 255)

# --- CLASSE DE ACELERAÇÃO (Webcam Leve) ---
class WebcamStream:
    def __init__(self, src=0):
        # Tenta DirectShow
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        
        # --- AJUSTE PARA HARDWARE BÁSICO ---
        # Não vamos forçar Full HD (1080p) pois consome muita RAM.
        # Vamos usar HD (1280x720) ou VGA (640x480) se ficar lento.
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
def tem_sobreposicao(box_corpo, box_epi):
    c_x1, c_y1, c_x2, c_y2 = map(int, box_corpo.xyxy[0])
    e_x1, e_y1, e_x2, e_y2 = map(int, box_epi.xyxy[0])

    inter_x1 = max(c_x1, e_x1)
    inter_y1 = max(c_y1, e_y1)
    inter_x2 = min(c_x2, e_x2)
    inter_y2 = min(c_y2, e_y2)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2: return True 
    return False

# --- DESENHO NA TELA ---
def processar_frame(frame, results):
    boxes = results[0].boxes
    
    pessoas, coletes, capacetes, cabecas = [], [], [], []
    rostos, oculos, maos, luvas, pes, botas = [], [], [], [], [], []

    # 1. Separação
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

    # --- LÓGICA 1: COLETE ---
    for pessoa in pessoas:
        tem_epi = False
        for epi in coletes:
            if tem_sobreposicao(pessoa, epi): tem_epi = True; break
        if not tem_epi: boxes_finais.append({'box': pessoa, 'cor': VERMELHO, 'msg': "SEM COLETE"})
    for x in coletes: boxes_finais.append({'box': x, 'cor': VERDE, 'msg': "OK: COLETE"})

    # --- LÓGICA 2: CAPACETE ---
    for cabeca in cabecas:
        tem_epi = False
        for epi in capacetes:
            if tem_sobreposicao(cabeca, epi): tem_epi = True; break
        if not tem_epi: boxes_finais.append({'box': cabeca, 'cor': VERMELHO, 'msg': "SEM CAPACETE"})
    for x in capacetes: boxes_finais.append({'box': x, 'cor': VERDE, 'msg': "OK: CAPACETE"})

    # --- LÓGICA 3: ÓCULOS ---
    for rosto in rostos:
        tem_epi = False
        for epi in oculos:
            if tem_sobreposicao(rosto, epi): tem_epi = True; break
        if not tem_epi: boxes_finais.append({'box': rosto, 'cor': VERMELHO, 'msg': "SEM OCULOS"})
    for x in oculos: boxes_finais.append({'box': x, 'cor': VERDE, 'msg': "OK: OCULOS"})

    # --- LÓGICA 4: LUVAS ---
    for mao in maos:
        tem_epi = False
        for epi in luvas:
            if tem_sobreposicao(mao, epi): tem_epi = True; break
        if not tem_epi: boxes_finais.append({'box': mao, 'cor': VERMELHO, 'msg': "SEM LUVA"})
    for x in luvas: boxes_finais.append({'box': x, 'cor': VERDE, 'msg': "OK: LUVA"})

    # --- LÓGICA 5: BOTAS ---
    for pe in pes:
        tem_epi = False
        for epi in botas:
            if tem_sobreposicao(pe, epi): tem_epi = True; break
        if not tem_epi: boxes_finais.append({'box': pe, 'cor': VERMELHO, 'msg': "SEM BOTA"})
    for x in botas: boxes_finais.append({'box': x, 'cor': VERDE, 'msg': "OK: BOTA"})

    # --- DESENHAR ---
    for item in boxes_finais:
        box = item['box']
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), item['cor'], 2)
        
        (w, h), _ = cv2.getTextSize(item['msg'], cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w + 5, y1), item['cor'], -1)
        
        cv2.putText(frame, item['msg'], (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, BRANCO if item['cor'] == VERMELHO else PRETO, 1)

    # Painel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (200, 80), PRETO, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"PESSOAS: {len(pessoas)}", (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, AMARELO, 1)
    cv2.putText(frame, f"OCULOS: {len(oculos)}", (10, 55), cv2.FONT_HERSHEY_DUPLEX, 0.6, BRANCO, 1)

    return frame

# --- PROGRAMA PRINCIPAL ---
print("🚀 Iniciando Sistema Otimizado (Celeron)...")
model = YOLO(ARQUIVO_MODELO)

print(f"📷 Câmera {CAMERA_INDEX}...")
webcam = WebcamStream(src=CAMERA_INDEX).start()
time.sleep(1.5)

if webcam.stopped:
    print("❌ Erro na câmera.")
    exit()

# TELA CHEIA
NOME_JANELA = "Monitoramento EPI - Leve"
cv2.namedWindow(NOME_JANELA, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(NOME_JANELA, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("✅ RODANDO! (Q para sair)")

frame_count = 0
ultimo_resultado = None

while True:
    frame = webcam.read()
    if frame is None: continue

    frame_count += 1

    # Pula frames para dar tempo do processador respirar
    if frame_count % PULAR_QUADROS == 0:
        results = model(frame, imgsz=TAMANHO_IMG_YOLO, conf=CONFIDENCE, verbose=False)
        frame_final = processar_frame(frame.copy(), results)
        ultimo_resultado = frame_final
    else:
        if ultimo_resultado is not None:
            frame_final = ultimo_resultado
        else:
            frame_final = frame

    cv2.imshow(NOME_JANELA, frame_final)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

webcam.stop()
cv2.destroyAllWindows()