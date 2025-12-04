import cv2
import time
from threading import Thread
from ultralytics import YOLO

# --- CONFIGURAÇÕES DE ALTA PERFORMANCE ---
ARQUIVO_MODELO = 'best.pt'
CONFIDENCE = 0.40           # Aumentei para 40% para evitar falsos positivos, já que a imagem está nítida
TAMANHO_IMG_YOLO = 640      # 640 é a resolução nativa do treino. Ideal para ver ÓCULOS e detalhes.
CAMERA_INDEX = 0            # 0 = Notebook, 1 = USB (Troque se necessário)

# --- NOMES REAIS ---
NOMES_CLASSES = {
    0: 'Pessoa', 1: 'Orelha', 2: 'Protetor Auricular', 3: 'Rosto', 
    4: 'Protetor Facial', 5: 'Mascara', 6: 'Pe', 7: 'Ferramenta', 
    8: 'Oculos', 9: 'Luvas', 10: 'Capacete', 11: 'Maos', 
    12: 'Cabeca', 13: 'Traje Medico', 14: 'Botas', 
    15: 'Traje Seguranca', 16: 'Colete'
}

# --- FILTRO ATIVO ---
ITENS_ATIVOS = [
    # Seguros (Verde)
    10, # Capacete
    16, # Colete
    8,  # Oculos (ATENÇÃO TOTAL AQUI)
    9,  # Luvas
    14, # Botas
    
    # Perigos (Vermelho - Avisar falta)
    12, # Cabeca
    3,  # Rosto
    11, # Maos
    6,  # Pe
]

VERDE = (0, 255, 0)
VERMELHO = (0, 0, 255)
AMARELO = (0, 255, 255)
PRETO = (0, 0, 0)
BRANCO = (255, 255, 255)

# --- CLASSE DE ACELERAÇÃO (Webcam FULL HD) ---
class WebcamStream:
    def __init__(self, src=0):
        # Tenta forçar o driver rápido (DSHOW)
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        
        # --- CONFIGURAÇÃO DE ALTA QUALIDADE ---
        # Tenta abrir em 1920x1080 (Full HD)
        # Se sua câmera não suportar, ela vai abrir na maior possível automaticamente
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Buffer pequeno para garantir tempo real (sem delay)
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

# --- LÓGICA DE SOBREPOSIÇÃO (QUEM ESTÁ USANDO O QUE) ---
def tem_sobreposicao(box_corpo, box_epi):
    c_x1, c_y1, c_x2, c_y2 = map(int, box_corpo.xyxy[0])
    e_x1, e_y1, e_x2, e_y2 = map(int, box_epi.xyxy[0])

    # Calcula intersecção
    inter_x1 = max(c_x1, e_x1)
    inter_y1 = max(c_y1, e_y1)
    inter_x2 = min(c_x2, e_x2)
    inter_y2 = min(c_y2, e_y2)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        return True 
    return False

# --- DESENHO NA TELA ---
def processar_frame(frame, results):
    boxes = results[0].boxes
    
    # Listas
    pessoas = []
    coletes = []
    capacetes = []
    cabecas = []
    
    # Listas para Óculos e Luvas
    rostos = []
    oculos = []
    maos = []
    luvas = []
    
    pes = []
    botas = []

    # 1. Separação Inteligente
    for box in boxes:
        cls = int(box.cls[0])
        if cls not in ITENS_ATIVOS and cls != 0: continue # Ignora o resto

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
        if not tem_epi: boxes_finais.append({'box': pessoa, 'cor': VERMELHO, 'msg': "SEM COLETE", 'espessura': 2})
    for x in coletes: boxes_finais.append({'box': x, 'cor': VERDE, 'msg': "OK: COLETE", 'espessura': 2})

    # --- LÓGICA 2: CAPACETE ---
    for cabeca in cabecas:
        tem_epi = False
        for epi in capacetes:
            if tem_sobreposicao(cabeca, epi): tem_epi = True; break
        if not tem_epi: boxes_finais.append({'box': cabeca, 'cor': VERMELHO, 'msg': "SEM CAPACETE", 'espessura': 2})
    for x in capacetes: boxes_finais.append({'box': x, 'cor': VERDE, 'msg': "OK: CAPACETE", 'espessura': 2})

    # --- LÓGICA 3: ÓCULOS (ATENÇÃO ESPECIAL) ---
    for rosto in rostos:
        tem_epi = False
        for epi in oculos:
            if tem_sobreposicao(rosto, epi): tem_epi = True; break
        
        if not tem_epi: 
            boxes_finais.append({'box': rosto, 'cor': VERMELHO, 'msg': "SEM OCULOS", 'espessura': 2})
    
    for x in oculos: 
        # Desenha ÓCULOS com linha mais grossa para destacar
        boxes_finais.append({'box': x, 'cor': VERDE, 'msg': "OK: OCULOS", 'espessura': 3})

    # --- LÓGICA 4: LUVAS ---
    for mao in maos:
        tem_epi = False
        for epi in luvas:
            if tem_sobreposicao(mao, epi): tem_epi = True; break
        if not tem_epi: boxes_finais.append({'box': mao, 'cor': VERMELHO, 'msg': "SEM LUVA", 'espessura': 2})
    for x in luvas: boxes_finais.append({'box': x, 'cor': VERDE, 'msg': "OK: LUVA", 'espessura': 2})

    # --- LÓGICA 5: BOTAS ---
    for pe in pes:
        tem_epi = False
        for epi in botas:
            if tem_sobreposicao(pe, epi): tem_epi = True; break
        if not tem_epi: boxes_finais.append({'box': pe, 'cor': VERMELHO, 'msg': "SEM BOTA", 'espessura': 2})
    for x in botas: boxes_finais.append({'box': x, 'cor': VERDE, 'msg': "OK: BOTA", 'espessura': 2})

    # --- DESENHAR TUDO ---
    for item in boxes_finais:
        box = item['box']
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), item['cor'], item['espessura'])
        
        # Fundo do texto mais bonito
        (w, h), _ = cv2.getTextSize(item['msg'], cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + w + 10, y1), item['cor'], -1)
        
        # Texto Branco para leitura perfeita
        cv2.putText(frame, item['msg'], (x1 + 5, y1 - 5), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, BRANCO, 1)

    # --- PAINEL INFORMATIVO SUPERIOR ESQUERDO ---
    # Fundo semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 100), PRETO, -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Informações
    qtd_pessoas = len(pessoas)
    qtd_oculos = len(oculos)
    
    cv2.putText(frame, f"PESSOAS: {qtd_pessoas}", (10, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, AMARELO, 1)
    
    # Status dos óculos no painel
    cor_oculos = VERDE if qtd_oculos >= qtd_pessoas and qtd_pessoas > 0 else VERMELHO
    if qtd_pessoas == 0: cor_oculos = BRANCO
    
    cv2.putText(frame, f"OCULOS DETECTADOS: {qtd_oculos}", (10, 75), cv2.FONT_HERSHEY_DUPLEX, 0.6, cor_oculos, 1)

    return frame

# --- PROGRAMA PRINCIPAL ---
print("🚀 Iniciando Sistema PRO (Alta Performance)...")
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
    results = model(frame, imgsz=TAMANHO_IMG_YOLO, conf=CONFIDENCE, verbose=False)
    frame_final = processar_frame(frame.copy(), results)

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