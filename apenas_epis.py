import cv2
import time
from threading import Thread
from ultralytics import YOLO

# --- CONFIGURAÇÕES ---
ARQUIVO_MODELO = 'best.pt'
CONFIDENCE = 0.25           # BAIXEI para pegar o Colete mais fácil
TAMANHO_IMG_YOLO = 320      # Mantive 320 para velocidade máxima
CAMERA_INDEX = 1            # 0 para Notebook, 1 para USB (ajuste conforme necessário)

# --- NOMES REAIS (SH17) ---
NOMES_CLASSES = {
    0: 'Pessoa', 1: 'Orelha', 2: 'Protetor Auricular', 3: 'Rosto', 
    4: 'Protetor Facial', 5: 'Mascara', 6: 'Pe', 7: 'Ferramenta', 
    8: 'Oculos', 9: 'Luvas', 10: 'Capacete', 11: 'Maos', 
    12: 'Cabeca', 13: 'Traje Medico', 14: 'Botas', 
    15: 'Traje Seguranca', 16: 'Colete'
}

# --- FILTRO DO QUE VAI APARECER NA TELA ---
ITENS_ATIVOS = [
    10, # Capacete
    16, # Colete
    8,  # Oculos
    9,  # Luvas
    14, # Botas
    12, # Cabeca (Perigo)
    3,  # Rosto (Perigo)
    11, # Maos (Perigo)
    6,  # Pe (Perigo)
]

VERDE = (0, 255, 0)
VERMELHO = (0, 0, 255)
AMARELO = (0, 255, 255)

# --- CLASSE DE ACELERAÇÃO (Webcam Otimizada) ---
class WebcamStream:
    def __init__(self, src=0):
        # Usa DirectShow no Windows para ser mais rápido
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        
        # --- O SEGREDO DA VELOCIDADE ---
        # Baixamos para 640x480. Isso alivia MUITO o processador.
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Buffersize 1 para garantir que pegamos sempre a imagem mais recente
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.stream.isOpened(): self.stopped = True
        else:
            (self.grabbed, self.frame) = self.stream.read()
            self.stopped = False

    def start(self):
        if self.stopped: return self
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped: return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self): return self.frame
    def stop(self): self.stopped = True

# --- FUNÇÃO DE LIMPEZA ---
def tem_sobreposicao(box1, box2):
    x1_min, y1_min, x1_max, y1_max = map(int, box1.xyxy[0])
    x2_min, y2_min, x2_max, y2_max = map(int, box2.xyxy[0])

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min: return False

    area_inter = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area_box1 = (x1_max - x1_min) * (y1_max - y1_min)
    
    if area_inter / area_box1 > 0.2: return True
    return False

# --- DESENHO INTELIGENTE ---
def desenhar_seguranca(frame, results):
    boxes = results[0].boxes
    seguros = []
    perigos = []
    
    ids_ok = [10, 16, 8, 9, 14] 
    ids_bad = [12, 3, 11, 6]

    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id not in ITENS_ATIVOS: continue 
        if cls_id in ids_ok: seguros.append(box)
        elif cls_id in ids_bad: perigos.append(box)

    perigos_reais = []
    for perigo in perigos:
        escondido = False
        cls_perigo = int(perigo.cls[0])
        for seguro in seguros:
            cls_seguro = int(seguro.cls[0])
            # Regras de anulação
            if cls_perigo == 12 and cls_seguro == 10: 
                if tem_sobreposicao(perigo, seguro): escondido = True
            if cls_perigo == 3 and cls_seguro == 8: 
                if tem_sobreposicao(perigo, seguro): escondido = True
            if cls_perigo == 11 and cls_seguro == 9: 
                if tem_sobreposicao(perigo, seguro): escondido = True
            if cls_perigo == 6 and cls_seguro == 14: 
                if tem_sobreposicao(perigo, seguro): escondido = True

        if not escondido: perigos_reais.append(perigo)

    todos = seguros + perigos_reais 

    for box in todos:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = NOMES_CLASSES.get(cls_id, 'Desconhecido')

        if cls_id in ids_bad:
            cor = VERMELHO
            msg = "PERIGO"
            if label == 'Cabeca': msg = "SEM CAPACETE"
            if label == 'Rosto': msg = "SEM OCULOS"
            if label == 'Maos': msg = "SEM LUVAS"
            if label == 'Pe': msg = "SEM BOTAS"
        else:
            cor = VERDE
            msg = f"OK: {label.upper()}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
        # Ajuste no tamanho da fonte para a tela menor (640x480)
        (w, h), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), cor, -1)
        cv2.putText(frame, msg, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255) if cor==VERMELHO else (0,0,0), 2)

    return frame

# --- PROGRAMA ---
print("🚀 Iniciando Detector Turbo (640x480)...")
model = YOLO(ARQUIVO_MODELO)

print(f"📷 Câmera {CAMERA_INDEX}...")
webcam = WebcamStream(src=CAMERA_INDEX).start()
time.sleep(1.0)

if webcam.stopped:
    print("❌ Erro na câmera.")
    exit()

fps_start = 0
while True:
    frame = webcam.read()
    if frame is None: continue

    # Aqui a mágica acontece: Imagem pequena + Confiança baixa = Rápido e Sensível
    results = model(frame, imgsz=TAMANHO_IMG_YOLO, conf=CONFIDENCE, verbose=False)
    frame_final = desenhar_seguranca(frame, results)

    # FPS
    fps_end = time.time()
    if (fps_end - fps_start) > 0: fps = 1 / (fps_end - fps_start)
    fps_start = fps_end
    cv2.putText(frame_final, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("EPI Turbo", frame_final)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

webcam.stop()
cv2.destroyAllWindows()