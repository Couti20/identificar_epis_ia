import cv2
import time
import numpy as np # Necessário para o cálculo IoU
from threading import Thread
from ultralytics import YOLO

# --- CONFIGURAÇÕES DE PRECISÃO ---
ARQUIVO_MODELO = 'principal.pt' # Ajustado para o nome que você mencionou
CONFIDENCE = 0.40 
TAMANHO_IMG_YOLO = 640 
CAMERA_INDEX = 0
IOU_LIMITE = 0.40           # NOVO: Mínimo de 40% de IoU para considerar um EPI 'em uso'.

# --- NOMES REAIS (Mapeamento de Mapeamento para Manutenção) ---
NOMES_CLASSES = {
    0: 'Pessoa', 1: 'Orelha', 2: 'Protetor Auricular', 3: 'Rosto', 
    4: 'Protetor Facial', 5: 'Mascara', 6: 'Pe', 7: 'Ferramenta', 
    8: 'Oculos', 9: 'Luvas', 10: 'Capacete', 11: 'Maos', 
    12: 'Cabeca', 13: 'Traje Medico', 14: 'Botas', 
    15: 'Traje Seguranca', 16: 'Colete'
}

# --- MAPEAMENTO DE ÍNDICES PARA FACILITAR A LEITURA DO CÓDIGO ---
# Usamos variáveis para evitar números mágicos (ex: 10, 12) no código
CL_PESSOA, CL_CABECA, CL_ROSTO, CL_MAOS, CL_PE = 0, 12, 3, 11, 6
CL_COLETE, CL_CAPACETE, CL_OCULOS, CL_LUVAS, CL_BOTAS = 16, 10, 8, 9, 14

ITENS_ATIVOS = [
    # Elementos de Conformidade
    CL_CAPACETE, CL_COLETE, CL_OCULOS, CL_LUVAS, CL_BOTAS, 
    # Elementos de Não-Conformidade (partes do corpo)
    CL_CABECA, CL_ROSTO, CL_MAOS, CL_PE, CL_PESSOA
]

VERDE = (0, 255, 0)
VERMELHO = (0, 0, 255)
AMARELO = (0, 255, 255)
PRETO = (0, 0, 0)
BRANCO = (255, 255, 255)

# --- CLASSE DE ACELERAÇÃO (WebcamStream omitida, mas mantida no código) ---
class WebcamStream:
    # ... (Seu código original de WebcamStream) ...
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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

# --- LÓGICA DE SOBREPOSIÇÃO MELHORADA (IoU) ---
def calcular_iou(box_corpo, box_epi, iou_limite=0.0):
    """
    Calcula o IoU (Intersection over Union) e verifica se ele atende ao limite.
    box_corpo e box_epi são tensores box.xyxy[0] do YOLO (x1, y1, x2, y2).
    """
    c_x1, c_y1, c_x2, c_y2 = map(int, box_corpo.xyxy[0])
    e_x1, e_y1, e_x2, e_y2 = map(int, box_epi.xyxy[0])

    # 1. Coordenadas da Intersecção
    inter_x1 = max(c_x1, e_x1)
    inter_y1 = max(c_y1, e_y1)
    inter_x2 = min(c_x2, e_x2)
    inter_y2 = min(c_y2, e_y2)

    # 2. Área da Intersecção
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    area_inter = inter_width * inter_height

    # 3. Área das Caixas
    area_corpo = (c_x2 - c_x1) * (c_y2 - c_y1)
    area_epi = (e_x2 - e_x1) * (e_y2 - e_y1)

    # 4. Cálculo IoU
    area_uniao = area_corpo + area_epi - area_inter
    
    # Evita divisão por zero
    iou = area_inter / area_uniao if area_uniao > 0 else 0.0
    
    return iou >= iou_limite


# --- DESENHO NA TELA ---
def processar_frame(frame, results):
    boxes = results[0].boxes
    
    # Dicionário para armazenar boxes por classe (melhor que listas)
    deteccoes = {cls: [] for cls in ITENS_ATIVOS + [CL_PESSOA]} 
    
    # 1. Separação Inteligente
    for box in boxes:
        cls = int(box.cls[0])
        if cls in deteccoes:
            deteccoes[cls].append(box)

    boxes_finais = [] 

    # --- LÓGICA 1: COLETE (Verifica PESSOA contra COLETE) ---
    for pessoa in deteccoes[CL_PESSOA]:
        tem_epi = False
        for colete in deteccoes[CL_COLETE]:
            # Usamos IoU, mas sem limite para Coletes (apenas sobreposição)
            if calcular_iou(pessoa, colete, iou_limite=0.01): tem_epi = True; break
        
        # Desenha a pessoa (ou a ausência de colete)
        if not tem_epi: boxes_finais.append({'box': pessoa, 'cor': VERMELHO, 'msg': "SEM COLETE", 'espessura': 2})
    for x in deteccoes[CL_COLETE]: boxes_finais.append({'box': x, 'cor': VERDE, 'msg': NOMES_CLASSES[CL_COLETE], 'espessura': 2})


    # --- LÓGICA 2: CAPACETE (Verifica CABEÇA contra CAPACETE) ---
    for cabeca in deteccoes[CL_CABECA]:
        tem_epi = False
        for capacete in deteccoes[CL_CAPACETE]:
            # **Aplica o IOU_LIMITE aqui para rigor**
            if calcular_iou(cabeca, capacete, IOU_LIMITE): tem_epi = True; break 
        
        if not tem_epi: boxes_finais.append({'box': cabeca, 'cor': VERMELHO, 'msg': "SEM CAPACETE", 'espessura': 2})
    for x in deteccoes[CL_CAPACETE]: boxes_finais.append({'box': x, 'cor': VERDE, 'msg': NOMES_CLASSES[CL_CAPACETE], 'espessura': 2})


    # --- LÓGICA 3: ÓCULOS (Verifica ROSTO contra ÓCULOS) ---
    for rosto in deteccoes[CL_ROSTO]:
        tem_epi = False
        for oculo in deteccoes[CL_OCULOS]:
            # O IoU é crítico para óculos, pois o rosto pode ser grande.
            if calcular_iou(rosto, oculo, IOU_LIMITE): tem_epi = True; break
        
        if not tem_epi: 
            boxes_finais.append({'box': rosto, 'cor': VERMELHO, 'msg': "SEM OCULOS", 'espessura': 2})
    for x in deteccoes[CL_OCULOS]: 
        boxes_finais.append({'box': x, 'cor': VERDE, 'msg': NOMES_CLASSES[CL_OCULOS], 'espessura': 3})


    # --- LÓGICA 4: LUVAS (Verifica MÃOS contra LUVAS) ---
    for mao in deteccoes[CL_MAOS]:
        tem_epi = False
        for luva in deteccoes[CL_LUVAS]:
            if calcular_iou(mao, luva, IOU_LIMITE): tem_epi = True; break
        if not tem_epi: boxes_finais.append({'box': mao, 'cor': VERMELHO, 'msg': "SEM LUVA", 'espessura': 2})
    for x in deteccoes[CL_LUVAS]: boxes_finais.append({'box': x, 'cor': VERDE, 'msg': NOMES_CLASSES[CL_LUVAS], 'espessura': 2})


    # --- LÓGICA 5: BOTAS (Verifica PÉS contra BOTAS) ---
    for pe in deteccoes[CL_PE]:
        tem_epi = False
        for bota in deteccoes[CL_BOTAS]:
            if calcular_iou(pe, bota, IOU_LIMITE): tem_epi = True; break
        if not tem_epi: boxes_finais.append({'box': pe, 'cor': VERMELHO, 'msg': "SEM BOTA", 'espessura': 2})
    for x in deteccoes[CL_BOTAS]: boxes_finais.append({'box': x, 'cor': VERDE, 'msg': NOMES_CLASSES[CL_BOTAS], 'espessura': 2})

    # --- DESENHAR TUDO ---
    for item in boxes_finais:
        box = item['box']
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), item['cor'], item['espessura'])
        
        (w, h), _ = cv2.getTextSize(item['msg'], cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + w + 10, y1), item['cor'], -1)
        
        cv2.putText(frame, item['msg'], (x1 + 5, y1 - 5), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, BRANCO, 1)

    # --- PAINEL INFORMATIVO SUPERIOR ESQUERDO ---
    # Fundo semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 100), PRETO, -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Informações
    qtd_pessoas = len(deteccoes[CL_PESSOA])
    # Total de itens de risco (não conformidade detectada)
    qtd_risco = len([item for item in boxes_finais if item['cor'] == VERMELHO])

    cv2.putText(frame, f"PESSOAS: {qtd_pessoas}", (10, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, AMARELO, 1)
    
    cor_risco = VERMELHO if qtd_risco > 0 else VERDE
    msg_risco = f"RISCO ATIVO: {qtd_risco}" if qtd_risco > 0 else "SEGURANCA OK"
    
    cv2.putText(frame, msg_risco, (10, 75), cv2.FONT_HERSHEY_DUPLEX, 0.6, cor_risco, 1)

    return frame

# --- PROGRAMA PRINCIPAL ---
print("🚀 Iniciando Sistema PRO (Alta Performance) com IoU rigoroso...")
model = YOLO(ARQUIVO_MODELO)

print(f"📷 Conectando Câmera {CAMERA_INDEX} em Full HD...")
webcam = WebcamStream(src=CAMERA_INDEX).start()
time.sleep(2.0)

if webcam.stopped:
    print("❌ Erro na câmera.")
    exit()

# TELA CHEIA
NOME_JANELA = "Monitoramento EPI - Alta Performance"
cv2.namedWindow(NOME_JANELA, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(NOME_JANELA, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("✅ RODANDO LISO! (Aperte 'q' para sair)")

fps_start = 0
while True:
    frame = webcam.read()
    if frame is None: continue

    # MODO LISO: Processa TODO quadro (sem pular) com resolução 640
    # O .copy() é crucial para que a thread da câmera continue rodando
    results = model(frame.copy(), imgsz=TAMANHO_IMG_YOLO, conf=CONFIDENCE, verbose=False)
    frame_final = processar_frame(frame, results)

    # FPS
    fps_end = time.time()
    if (fps_end - fps_start) > 0: fps = 1 / (fps_end - fps_start)
    fps_start = fps_end
    
    # Mostra FPS no canto inferior direito discreto
    h, w, _ = frame_final.shape
    cv2.putText(frame_final, f"FPS: {int(fps)}", (w - 150, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BRANCO, 1)

    cv2.imshow(NOME_JANELA, frame_final)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.stop()
cv2.destroyAllWindows()