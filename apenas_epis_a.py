import cv2
import time
from threading import Thread
from ultralytics import YOLO

# --- CONFIGURAÇÕES ---
ARQUIVO_MODELO = 'best.pt'
CONFIDENCE = 0.30
TAMANHO_IMG_YOLO = 192      # Baixei para 192 (Super Pixelado para a IA, mas ultrarrápido)
CAMERA_INDEX = 0            # 0=Notebook, 1=USB
PULAR_QUADROS = 3           # Analisa apenas 1 a cada 3 frames (Alivia CPU)

# --- NOMES REAIS ---
NOMES_CLASSES = {
    0: 'Pessoa', 1: 'Orelha', 2: 'Protetor Auricular', 3: 'Rosto', 
    4: 'Protetor Facial', 5: 'Mascara', 6: 'Pe', 7: 'Ferramenta', 
    8: 'Oculos', 9: 'Luvas', 10: 'Capacete', 11: 'Maos', 
    12: 'Cabeca', 13: 'Traje Medico', 14: 'Botas', 
    15: 'Traje Seguranca', 16: 'Colete'
}

VERDE = (0, 255, 0)
VERMELHO = (0, 0, 255)
AMARELO = (0, 255, 255)

# --- CLASSE DE ACELERAÇÃO (Webcam 320x240) ---
class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        # Resolução Baixa (QVGA) para velocidade máxima
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
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

# --- FUNÇÃO QUE CRUZA DADOS (Pessoa x Colete) ---
def tem_sobreposicao(box_pessoa, box_colete):
    p_x1, p_y1, p_x2, p_y2 = map(int, box_pessoa.xyxy[0])
    c_x1, c_y1, c_x2, c_y2 = map(int, box_colete.xyxy[0])

    # Verifica intersecção
    inter_x1 = max(p_x1, c_x1)
    inter_y1 = max(p_y1, c_y1)
    inter_x2 = min(p_x2, c_x2)
    inter_y2 = min(p_y2, c_y2)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        return True # Estão se tocando
    return False

# --- DESENHO INTELIGENTE ---
def processar_frame(frame, results):
    boxes = results[0].boxes
    
    # Listas temporárias
    pessoas = []
    coletes = []
    capacetes = []
    cabecas = []
    outros_seguros = [] # Oculos, Luvas, Botas
    outros_perigos = [] # Maos, Rosto, Pe

    # 1. Separar o que a IA viu
    for box in boxes:
        cls = int(box.cls[0])
        
        if cls == 0: pessoas.append(box)          # Pessoa
        elif cls == 16: coletes.append(box)       # Colete
        elif cls == 10: capacetes.append(box)     # Capacete
        elif cls == 12: cabecas.append(box)       # Cabeca
        
        elif cls in [8, 9, 14]: outros_seguros.append(box) # Oculos, Luvas, Botas
        elif cls in [3, 11, 6]: outros_perigos.append(box) # Rosto, Maos, Pe

    boxes_finais = [] # Lista do que vamos desenhar

    # 2. Lógica do COLETE (Pessoa vs Colete)
    for pessoa in pessoas:
        tem_colete = False
        for colete in coletes:
            if tem_sobreposicao(pessoa, colete):
                tem_colete = True
                break # Achou colete, passa para a próxima pessoa
        
        if not tem_colete:
            # Se a pessoa NÃO tem colete, desenha ela em VERMELHO
            boxes_finais.append({'box': pessoa, 'cor': VERMELHO, 'msg': "SEM COLETE"})
    
    # Adiciona os Coletes encontrados (Verde)
    for colete in coletes:
        boxes_finais.append({'box': colete, 'cor': VERDE, 'msg': "OK: COLETE"})

    # 3. Lógica do CAPACETE (Mantendo a anterior)
    for cabeca in cabecas:
        protegida = False
        for capacete in capacetes:
            if tem_sobreposicao(cabeca, capacete):
                protegida = True
        if not protegida:
            boxes_finais.append({'box': cabeca, 'cor': VERMELHO, 'msg': "SEM CAPACETE"})

    for capacete in capacetes:
        boxes_finais.append({'box': capacete, 'cor': VERDE, 'msg': "OK: CAPACETE"})

    # 4. Outros itens
    for item in outros_seguros:
        label = NOMES_CLASSES[int(item.cls[0])]
        boxes_finais.append({'box': item, 'cor': VERDE, 'msg': f"OK: {label.upper()}"})
    
    for item in outros_perigos:
        label = NOMES_CLASSES[int(item.cls[0])]
        msg = "SEM PROTECAO"
        if label == 'Maos': msg = "SEM LUVAS"
        elif label == 'Rosto': msg = "SEM OCULOS"
        elif label == 'Pe': msg = "SEM BOTAS"
        boxes_finais.append({'box': item, 'cor': VERMELHO, 'msg': msg})

    # 5. Desenhar tudo na tela
    for item in boxes_finais:
        box = item['box']
        cor = item['cor']
        msg = item['msg']
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
        cv2.putText(frame, msg, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

    return frame

# --- PROGRAMA ---
print("🚀 Iniciando Detector Ultra Leve...")
model = YOLO(ARQUIVO_MODELO)

print(f"📷 Câmera {CAMERA_INDEX}...")
webcam = WebcamStream(src=CAMERA_INDEX).start()
time.sleep(1.0)

frame_count = 0
ultimo_resultado = None # Guarda o desenho anterior para não piscar

while True:
    frame = webcam.read()
    if frame is None: continue

    frame_count += 1

    # --- PULAR QUADROS (O Segredo da Velocidade) ---
    # Só processa a IA 1 vez a cada 3 frames. Nos outros, mostra o vídeo puro.
    if frame_count % PULAR_QUADROS == 0:
        results = model(frame, imgsz=TAMANHO_IMG_YOLO, conf=CONFIDENCE, verbose=False)
        # Desenha e guarda o frame pintado na memória
        frame_pintado = processar_frame(frame.copy(), results)
        ultimo_resultado = frame_pintado
    else:
        # Nos frames que pulamos, mostramos o último resultado da IA
        # para não ficar "piscando", ou mostramos o frame atual se quiser fluidez total
        if ultimo_resultado is not None:
            # Mistura levemente para parecer fluido
            frame_pintado = ultimo_resultado
        else:
            frame_pintado = frame

    cv2.imshow("Monitoramento EPI (Modo Turbo)", frame_pintado)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

webcam.stop()
cv2.destroyAllWindows()