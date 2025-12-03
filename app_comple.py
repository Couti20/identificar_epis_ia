import cv2
import time
from threading import Thread
from ultralytics import YOLO

# --- CONFIGURAÇÕES ---
ARQUIVO_MODELO = 'best.pt'
CONFIDENCE = 0.35
TAMANHO_IMG_YOLO = 320
CAMERA_INDEX = 1            # Câmera USB

# --- NOMES REAIS ---
NOMES_CLASSES = {
    0: 'Pessoa', 1: 'Orelha', 2: 'Protetor Auricular', 3: 'Rosto', 
    4: 'Protetor Facial', 5: 'Mascara', 6: 'Pe', 7: 'Ferramenta', 
    8: 'Oculos', 9: 'Luvas', 10: 'Capacete', 11: 'Maos', 
    12: 'Cabeca', 13: 'Traje Medico', 14: 'Sapatos', 
    15: 'Traje Seguranca', 16: 'Colete'
}

VERDE = (0, 255, 0)
VERMELHO = (0, 0, 255)
AMARELO = (0, 255, 255)

# --- CLASSE DE ACELERAÇÃO (Webcam) ---
class WebcamStream:
    def __init__(self, src=1):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.stream.isOpened():
            self.stopped = True
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

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# --- FUNÇÃO AUXILIAR: Verifica se dois quadrados se sobrepõem ---
def tem_sobreposicao(box1, box2):
    # Coordenadas do Box 1 (Perigo)
    x1_min, y1_min, x1_max, y1_max = map(int, box1.xyxy[0])
    # Coordenadas do Box 2 (Seguro)
    x2_min, y2_min, x2_max, y2_max = map(int, box2.xyxy[0])

    # Calcula a intersecção (área onde eles se cruzam)
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return False # Não se tocam

    # Se a área de intersecção for grande, consideramos sobreposição
    area_inter = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area_box1 = (x1_max - x1_min) * (y1_max - y1_min)
    
    # Se mais de 30% da "Cabeça" estiver coberta pelo "Capacete", esconda a cabeça
    if area_inter / area_box1 > 0.3:
        return True
    return False

# --- FUNÇÃO DE DESENHO INTELIGENTE (COM FILTRO) ---
def desenhar_seguranca(frame, results):
    boxes = results[0].boxes
    
    # Separa o que é Seguro e o que é Perigo
    seguros = []
    perigos = []
    outros = []

    # Lista de IDs que anulam o perigo (Ex: Capacete anula Cabeça)
    # 10=Capacete, 9=Luvas, 8=Oculos, 5=Mascara, 2=Prot.Auricular
    ids_seguros = [10, 9, 8, 5, 4, 2, 16, 15] 
    
    # IDs de Perigo
    # 12=Cabeca, 11=Maos, 3=Rosto, 1=Orelha
    ids_perigos = [12, 11, 3, 1]

    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id in ids_seguros:
            seguros.append(box)
        elif cls_id in ids_perigos:
            perigos.append(box)
        else:
            outros.append(box)

    # --- O FILTRO MÁGICO ---
    # Só desenha o perigo se NÃO houver um item seguro em cima dele
    perigos_reais = []
    for perigo in perigos:
        escondido = False
        cls_perigo = int(perigo.cls[0])
        
        for seguro in seguros:
            cls_seguro = int(seguro.cls[0])
            
            # Regra: Capacete (10) esconde Cabeça (12)
            if cls_perigo == 12 and cls_seguro == 10:
                if tem_sobreposicao(perigo, seguro): escondido = True
            
            # Regra: Luvas (9) esconde Mãos (11)
            if cls_perigo == 11 and cls_seguro == 9:
                if tem_sobreposicao(perigo, seguro): escondido = True

            # Regra: Óculos (8) ou Máscara (5) esconde Rosto (3)
            if cls_perigo == 3 and cls_seguro in [8, 5, 4]:
                if tem_sobreposicao(perigo, seguro): escondido = True

        if not escondido:
            perigos_reais.append(perigo)

    # Agora desenha tudo (Seguros + Perigos Reais + Outros)
    todos_boxes = seguros + perigos_reais + outros

    for box in todos_boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = NOMES_CLASSES.get(cls_id, 'Desconhecido')

        # Define cor e mensagem
        if cls_id in ids_perigos: # Se sobrou aqui, é perigo mesmo
            cor = VERMELHO
            msg = f"SEM PROTECAO!"
            if label == 'Cabeca': msg = "SEM CAPACETE"
            if label == 'Maos': msg = "SEM LUVAS"
            if label == 'Rosto': msg = "SEM OCULOS"
        elif cls_id in ids_seguros:
            cor = VERDE
            msg = f"OK: {label}"
        else:
            cor = AMARELO
            msg = label

        # Desenha
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
        # Fundo do texto para ler melhor
        (w, h), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), cor, -1)
        cv2.putText(frame, f"{msg}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0) if cor!=VERMELHO else (255,255,255), 2)

    return frame

# --- LOOP PRINCIPAL ---
print("🚀 Iniciando Sistema Inteligente...")
model = YOLO(ARQUIVO_MODELO)

print(f"📷 Tentando Câmera {CAMERA_INDEX}...")
webcam = WebcamStream(src=CAMERA_INDEX).start()
time.sleep(2.0)

if webcam.stopped:
    print("❌ Erro na câmera. Verifique USB ou mude CAMERA_INDEX.")
    exit()

print("✅ Câmera OK! (Q para sair)")

fps_start = 0
while True:
    frame = webcam.read()
    if frame is None: continue

    results = model(frame, imgsz=TAMANHO_IMG_YOLO, conf=CONFIDENCE, verbose=False)
    
    # Chama a nova função de desenho com filtro
    frame_final = desenhar_seguranca(frame, results)

    # FPS
    fps_end = time.time()
    if (fps_end - fps_start) > 0: fps = 1 / (fps_end - fps_start)
    fps_start = fps_end
    cv2.putText(frame_final, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Monitoramento Inteligente EPI", frame_final)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

webcam.stop()
cv2.destroyAllWindows()