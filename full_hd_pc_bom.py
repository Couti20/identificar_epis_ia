import cv2
import time
from threading import Thread
from ultralytics import YOLO

# --- CONFIGURAÇÕES DE ALTA PERFORMANCE ---
ARQUIVO_MODELO = 'principal.pt'
CONFIDENCE = 0.40           # Limiar de confiança para detecção
TAMANHO_IMG_YOLO = 640      # Resolução de entrada para o modelo
CAMERA_INDEX = 0            # 0 = Notebook, 1 = USB

# --- NOMES REAIS ---
NOMES_CLASSES = {
    0: 'Pessoa', 1: 'Orelha', 2: 'Protetor Auricular', 3: 'Rosto', 
    4: 'Protetor Facial', 5: 'Mascara', 6: 'Pe', 7: 'Ferramenta', 
    8: 'Oculos', 9: 'Luvas', 10: 'Capacete', 11: 'Maos', 
    12: 'Cabeca', 13: 'Traje Medico', 14: 'Botas', 
    15: 'Traje Seguranca', 16: 'Colete'
}

# --- FILTRO ATIVO (EPIS e Partes do Corpo) ---
ITENS_ATIVOS = [
    # Seguros (Verde)
    10, # Capacete
    16, # Colete
    8,  # Oculos
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

# --- LÓGICA DE SOBREPOSIÇÃO (QUEM ESTÁ USANDO O QUE) ---
def tem_sobreposicao(box_corpo, box_epi):
    """Verifica se a caixa do corpo tem sobreposição com a caixa do EPI."""
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
    
    # Listas para detecções
    pessoas, coletes, capacetes, cabecas = [], [], [], []
    rostos, oculos, maos, luvas = [], [], [], []
    pes, botas = [], []
    
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
    
    # --- NOVO: STATUS GERAL DE EPIs (Para o painel de resumo) ---
    status_ep_resumo = {
        'Capacete': {'corpo': cabecas, 'epi': capacetes, 'nome_epi': 'CAPACETE'},
        'Colete': {'corpo': pessoas, 'epi': coletes, 'nome_epi': 'COLETE'},
        'Oculos': {'corpo': rostos, 'epi': oculos, 'nome_epi': 'OCULOS'},
        'Luvas': {'corpo': maos, 'epi': luvas, 'nome_epi': 'LUVA'},
        'Botas': {'corpo': pes, 'epi': botas, 'nome_epi': 'BOTA'},
    }
    
    # Inicializa o scorecard
    scorecard = {}
    
    # --- LÓGICA DE DETECÇÃO E SOBREPOSIÇÃO (Desenho e Scorecard) ---
    
    for item_nome, listas in status_ep_resumo.items():
        corpo_list = listas['corpo']
        epi_list = listas['epi']
        nome_epi = listas['nome_epi']
        
        # 1. Verifica se a parte do corpo está coberta
        for corpo_box in corpo_list:
            is_covered = False
            for epi_box in epi_list:
                if tem_sobreposicao(corpo_box, epi_box):
                    is_covered = True
                    break
            
            # Se não está coberto, é FALTA (VERMELHO)
            if not is_covered: 
                boxes_finais.append({'box': corpo_box, 'cor': VERMELHO, 'msg': f"SEM {nome_epi}", 'espessura': 2})
            else:
                # Se está coberto, desenha o EPI como OK (VERDE)
                # NOTA: Desenhar o EPI diretamente aqui garante que o EPI coberto seja verde
                # O loop for epi_list abaixo garante que os EPIs não sobrepostos também sejam desenhados (se houver).
                pass 
        
        # 2. Desenha todos os EPIs detectados (Se sobrepostos ou não, ficam VERDES)
        for epi_box in epi_list:
            # Óculos é desenhado mais grosso para destaque
            espessura = 3 if item_nome == 'Oculos' else 2
            boxes_finais.append({'box': epi_box, 'cor': VERDE, 'msg': f"OK: {nome_epi}", 'espessura': espessura})

        # 3. Atualiza o Scorecard (Verificação Simples: Há partes do corpo sem o EPI?)
        
        # Encontra todas as partes do corpo que ESTÃO DESCOBERTAS
        partes_descobertas = 0
        for corpo_box in corpo_list:
            is_covered = False
            for epi_box in epi_list:
                if tem_sobreposicao(corpo_box, epi_box):
                    is_covered = True
                    break
            if not is_covered:
                partes_descobertas += 1
                
        # Status final para o painel
        total_corpo = len(corpo_list)
        if total_corpo == 0:
            scorecard[item_nome] = {'status': 'NÃO DETECTADO', 'cor': BRANCO}
        elif partes_descobertas > 0:
            scorecard[item_nome] = {'status': 'FALTA', 'cor': VERMELHO}
        else:
            scorecard[item_nome] = {'status': 'OK', 'cor': VERDE}
    
    # --- DESENHAR CAIXAS E MENSAGENS ---
    for item in boxes_finais:
        box = item['box']
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), item['cor'], item['espessura'])
        
        # Fundo do texto
        (w_txt, h_txt), _ = cv2.getTextSize(item['msg'], cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + w_txt + 10, y1), item['cor'], -1)
        
        # Texto Branco
        cv2.putText(frame, item['msg'], (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, BRANCO, 1)

    # --- PAINEL DE STATUS RESUMIDO (Lateral Direita) ---
    h, w, _ = frame.shape
    largura_painel = 320
    
    # Fundo do Painel Lateral
    overlay_painel = frame.copy()
    cv2.rectangle(overlay_painel, (w - largura_painel, 0), (w, h), PRETO, -1)
    cv2.addWeighted(overlay_painel, 0.7, frame, 1 - 0.7, 0, frame)

    # Título
    cv2.putText(frame, "STATUS EPI (Pessoa)", (w - largura_painel + 20, 35), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, AMARELO, 1)
    
    y_offset = 70
    
    # Itera sobre o Scorecard para exibir o resumo
    for item_nome, status_data in scorecard.items():
        y_offset += 35
        status_msg = status_data['status']
        status_cor = status_data['cor']
        
        cv2.putText(frame, f"- {item_nome}:", (w - largura_painel + 20, y_offset), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, BRANCO, 1)
        
        cv2.putText(frame, status_msg, (w - largura_painel + 170, y_offset), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, status_cor, 1)
        
    # --- PAINEL INFORMATIVO SUPERIOR ESQUERDO (Total de Pessoas/Óculos) ---
    
    overlay_topo = frame.copy()
    cv2.rectangle(overlay_topo, (0, 0), (280, 100), PRETO, -1)
    cv2.addWeighted(overlay_topo, 0.7, frame, 1 - 0.7, 0, frame)

    qtd_pessoas = len(pessoas)
    qtd_oculos = len(oculos)
    
    cv2.putText(frame, f"PESSOAS: {qtd_pessoas}", (10, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, AMARELO, 1)
    
    cor_oculos = VERDE if qtd_pessoas > 0 and scorecard.get('Oculos', {}).get('status') == 'OK' else VERMELHO
    if qtd_pessoas == 0: cor_oculos = BRANCO
    
    cv2.putText(frame, f"OCULOS (Total): {qtd_oculos}", (10, 75), cv2.FONT_HERSHEY_DUPLEX, 0.6, cor_oculos, 1)
    
    return frame

# --- PROGRAMA PRINCIPAL ---
print("🚀 Iniciando Sistema PRO (Alta Performance)...")
model = YOLO(ARQUIVO_MODELO)

print(f"📷 Conectando Câmera {CAMERA_INDEX} em Full HD...")
webcam = WebcamStream(src=CAMERA_INDEX).start()
time.sleep(2.0)

if webcam.stopped:
    print("❌ Erro na câmera. Verifique a conexão.")
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

    # MODO LISO: Processa TODO quadro com resolução 640
    results = model(frame, imgsz=TAMANHO_IMG_YOLO, conf=CONFIDENCE, verbose=False)
    frame_final = processar_frame(frame.copy(), results)

    # FPS
    fps_end = time.time()
    fps = 0
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