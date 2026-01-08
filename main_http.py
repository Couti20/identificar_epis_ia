"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     SISTEMA DE DETECÇÃO DE EPIs v2.0                         ║
║                    Monitoramento de Segurança em Tempo Real                  ║
║                        VERSÃO CÂMERA HTTP (CELULAR)                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Autor: Sistema de IA para Segurança do Trabalho                             ║
║  Modelo: YOLOv8/v11 - Treinado com 83% de precisão                           ║
║  Câmera: Stream HTTP (IP Webcam, DroidCam, etc)                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

INSTRUÇÕES:
1. Instale o app "IP Webcam" no celular (Android)
2. Abra o app e clique em "Iniciar servidor"
3. Anote o endereço que aparece (ex: http://192.168.1.100:8080)
4. Cole o endereço na variável CAMERA_URL abaixo
5. Execute: python main_http.py
"""

import cv2
import time
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════════════════
#                         CONFIGURAÇÃO DA CÂMERA HTTP
# ══════════════════════════════════════════════════════════════════════════════

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  COLE O ENDEREÇO DA SUA CÂMERA AQUI:                                        │
# │                                                                             │
# │  Exemplos:                                                                  │
# │  - IP Webcam:  "http://192.168.1.100:8080/video"                           │
# │  - DroidCam:   "http://192.168.1.100:4747/video"                           │
# │  - Iriun:      "http://192.168.1.100:8080/video"                           │
# │  - RTSP:       "rtsp://192.168.1.100:8554/live"                            │
# │                                                                             │
# │  Para webcam normal, use:                                                   │
# │  - Webcam integrada: 0                                                      │
# │  - USB externa:      1                                                      │
# └─────────────────────────────────────────────────────────────────────────────┘

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ESCOLHA O TIPO DE CÂMERA E CONFIGURE ABAIXO:                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# OPÇÃO 1: DroidCam / IP Webcam (celular)
CAMERA_IP = "192.168.137.11"
CAMERA_PORTA = 4747

# OPÇÃO 2: Câmera de Segurança (RTSP) - descomente e configure se usar
# CAMERA_URL_RTSP = "rtsp://admin:senha@192.168.1.100:554/stream1"

# ═══════════════════════════════════════════════════════════════════════════
# NÃO ALTERE ABAIXO (configuração automática)
# ═══════════════════════════════════════════════════════════════════════════

# Verifica se tem RTSP configurado
try:
    CAMERA_URL_RTSP
    CAMERA_URLS = [CAMERA_URL_RTSP]  # Usa RTSP direto
except NameError:
    # Usa HTTP (DroidCam/IP Webcam)
    CAMERA_URLS = [
        f"http://{CAMERA_IP}:{CAMERA_PORTA}/video",
        f"http://{CAMERA_IP}:{CAMERA_PORTA}/mjpegfeed",
        f"http://{CAMERA_IP}:{CAMERA_PORTA}/videofeed",
        f"http://{CAMERA_IP}:{CAMERA_PORTA}",
    ]
CAMERA_URL = None  # Será definido automaticamente

# Outras configurações
MODELO_PATH = 'oitenta-tres.pt'
CONFIANCA = 0.40
TAMANHO_INFERENCIA = 640
IOU_LIMITE = 0.40


# ══════════════════════════════════════════════════════════════════════════════
#                          MAPEAMENTO DE CLASSES
# ══════════════════════════════════════════════════════════════════════════════

NOMES_CLASSES = {
    0: 'Pessoa', 1: 'Orelha', 2: 'Protetor Auricular', 3: 'Rosto',
    4: 'Protetor Facial', 5: 'Máscara', 6: 'Pé', 7: 'Ferramenta',
    8: 'Óculos', 9: 'Luvas', 10: 'Capacete', 11: 'Mãos',
    12: 'Cabeça', 13: 'Traje Médico', 14: 'Botas',
    15: 'Traje Segurança', 16: 'Colete'
}

# IDs das classes
CL_PESSOA, CL_CABECA, CL_ROSTO, CL_MAOS, CL_PE = 0, 12, 3, 11, 6
CL_COLETE, CL_CAPACETE, CL_OCULOS, CL_LUVAS, CL_BOTAS = 16, 10, 8, 9, 14

EPIS = [CL_CAPACETE, CL_COLETE, CL_OCULOS, CL_LUVAS, CL_BOTAS]
CORPO = [CL_CABECA, CL_ROSTO, CL_MAOS, CL_PE]
MONITORADOS = EPIS + CORPO + [CL_PESSOA]

# Regras: parte do corpo -> EPI correspondente
REGRAS = {
    CL_CABECA: CL_CAPACETE,
    CL_ROSTO: CL_OCULOS,
    CL_MAOS: CL_LUVAS,
    CL_PE: CL_BOTAS,
}

ALERTAS = {
    CL_CABECA: "SEM CAPACETE",
    CL_ROSTO: "SEM OCULOS",
    CL_MAOS: "SEM LUVAS",
    CL_PE: "SEM BOTAS",
    CL_PESSOA: "SEM COLETE",
}

# Cores BGR
VERDE = (0, 255, 0)
VERMELHO = (0, 0, 255)
AMARELO = (0, 255, 255)
PRETO = (0, 0, 0)
BRANCO = (255, 255, 255)


# ══════════════════════════════════════════════════════════════════════════════
#                              FUNÇÕES
# ══════════════════════════════════════════════════════════════════════════════

def calcular_iou(box1, box2) -> float:
    """Calcula IoU entre duas caixas."""
    x1_1, y1_1, x2_1, y2_1 = map(int, box1.xyxy[0])
    x1_2, y1_2, x2_2, y2_2 = map(int, box2.xyxy[0])
    
    inter_x1, inter_y1 = max(x1_1, x1_2), max(y1_1, y1_2)
    inter_x2, inter_y2 = min(x2_1, x2_2), min(y2_1, y2_2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    area_inter = inter_w * inter_h
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    area_uniao = area1 + area2 - area_inter
    
    return area_inter / area_uniao if area_uniao > 0 else 0.0


def tem_epi(box_corpo, lista_epis, iou_limite):
    """Verifica se algum EPI cobre a parte do corpo."""
    for epi in lista_epis:
        if calcular_iou(box_corpo, epi) >= iou_limite:
            return True
    return False


def processar_frame(frame, results):
    """Processa detecções e retorna frame anotado."""
    boxes = results[0].boxes
    
    # Separa detecções por classe
    deteccoes = {cls: [] for cls in MONITORADOS}
    for box in boxes:
        cls = int(box.cls[0])
        if cls in deteccoes:
            deteccoes[cls].append(box)
    
    anotacoes = []
    
    # 1. COLETE
    for pessoa in deteccoes[CL_PESSOA]:
        if not tem_epi(pessoa, deteccoes[CL_COLETE], 0.01):
            anotacoes.append({'box': pessoa, 'cor': VERMELHO, 'msg': ALERTAS[CL_PESSOA]})
    for colete in deteccoes[CL_COLETE]:
        anotacoes.append({'box': colete, 'cor': VERDE, 'msg': NOMES_CLASSES[CL_COLETE]})
    
    # 2. DEMAIS EPIs
    for parte_corpo, epi_classe in REGRAS.items():
        for box_corpo in deteccoes[parte_corpo]:
            if not tem_epi(box_corpo, deteccoes[epi_classe], IOU_LIMITE):
                anotacoes.append({'box': box_corpo, 'cor': VERMELHO, 'msg': ALERTAS[parte_corpo]})
        for epi in deteccoes[epi_classe]:
            anotacoes.append({'box': epi, 'cor': VERDE, 'msg': NOMES_CLASSES[epi_classe]})
    
    # DESENHA
    for item in anotacoes:
        box = item['box']
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), item['cor'], 2)
        
        (w, h), _ = cv2.getTextSize(item['msg'], cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + w + 10, y1), item['cor'], -1)
        cv2.putText(frame, item['msg'], (x1 + 5, y1 - 5), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, BRANCO, 1)
    
    # PAINEL
    qtd_pessoas = len(deteccoes[CL_PESSOA])
    qtd_riscos = sum(1 for a in anotacoes if a['cor'] == VERMELHO)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 100), PRETO, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, f"PESSOAS: {qtd_pessoas}", (10, 35), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, AMARELO, 1)
    
    if qtd_riscos > 0:
        cv2.putText(frame, f"RISCO ATIVO: {qtd_riscos}", (10, 75), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, VERMELHO, 1)
    else:
        cv2.putText(frame, "SEGURANCA OK", (10, 75), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, VERDE, 1)
    
    return frame


# ══════════════════════════════════════════════════════════════════════════════
#                            PROGRAMA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def conectar_camera():
    """Tenta conectar usando todas as URLs possíveis."""
    print(f"\n[2/3] Conectando câmera...")
    
    for url in CAMERA_URLS:
        print(f"      Testando: {url}")
        cap = cv2.VideoCapture(url)
        time.sleep(1)  # Aguarda conexão
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"      ✓ Conectado via: {url}")
                return cap, url
            cap.release()
    
    return None, None


def main():
    global CAMERA_URL
    print("=" * 60)
    print("   SISTEMA DE DETECÇÃO DE EPIs - CÂMERA HTTP")
    print("=" * 60)
    
    # Carrega modelo
    print(f"\n[1/3] Carregando modelo: {MODELO_PATH}")
    try:
        modelo = YOLO(MODELO_PATH)
        print("      ✓ Modelo carregado!")
    except Exception as e:
        print(f"      ✗ Erro: {e}")
        return
    
    # Conecta câmera (tenta todas as URLs)
    cap, CAMERA_URL = conectar_camera()
    
    if cap is None:
        print("      ✗ Erro: Não foi possível conectar à câmera!")
        print("\n      Verifique:")
        print("      - A câmera está ligada e na mesma rede?")
        print("      - O IP e porta estão corretos?")
        print("      - Para RTSP: usuário e senha estão corretos?")
        return
    
    print("      ✓ Câmera conectada!")
    
    # Pega resolução
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"      Resolução: {largura}x{altura}")
    
    # Janela
    print("\n[3/3] Iniciando monitoramento...")
    NOME_JANELA = "EPI Monitor - Camera HTTP (Q para sair)"
    cv2.namedWindow(NOME_JANELA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(NOME_JANELA, 1280, 720)
    
    print("\n" + "=" * 60)
    print("   ✓ SISTEMA ATIVO - Pressione 'Q' para sair")
    print("=" * 60 + "\n")
    
    fps_inicio = time.time()
    fps = 0
    frames_perdidos = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                frames_perdidos += 1
                if frames_perdidos > 30:
                    print("[!] Conexão perdida, tentando reconectar...")
                    cap.release()
                    cap = cv2.VideoCapture(CAMERA_URL)
                    frames_perdidos = 0
                continue
            
            frames_perdidos = 0
            
            # Inferência
            results = modelo(
                frame.copy(),
                imgsz=TAMANHO_INFERENCIA,
                conf=CONFIANCA,
                verbose=False
            )
            
            # Processa
            frame_final = processar_frame(frame, results)
            
            # FPS
            fps_fim = time.time()
            if fps_fim - fps_inicio > 0:
                fps = int(1 / (fps_fim - fps_inicio))
            fps_inicio = fps_fim
            
            h, w = frame_final.shape[:2]
            cv2.putText(frame_final, f"FPS: {fps}", (w - 150, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, BRANCO, 1)
            cv2.putText(frame_final, "HTTP", (w - 150, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, AMARELO, 1)
            
            cv2.imshow(NOME_JANELA, frame_final)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n[!] Interrompido pelo usuário")
    
    finally:
        print("\n[*] Encerrando...")
        cap.release()
        cv2.destroyAllWindows()
        print("[*] Sistema encerrado!")


if __name__ == "__main__":
    main()
