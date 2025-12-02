import cv2
import threading
import time
import math
import numpy as np
from ultralytics import YOLO

class DetectorEPIEstavel:
    """
    Versão Estável e TELA CHEIA:
    - Força a imagem a esticar para 1920x1080 (Full HD).
    - Janela abre em modo Fullscreen (sem bordas).
    """

    # --- CONFIGURAÇÕES VISUAIS ---
    COR_PERIGO = (0, 0, 255)    # Vermelho
    COR_SEGURO = (0, 255, 0)    # Verde
    COR_NEUTRO = (255, 200, 0)  # Azul Claro

    # --- CONFIGURAÇÃO DE PERSISTÊNCIA ---
    TEMPO_MEMORIA = 0.8 
    DISTANCIA_MAX_MATCH = 150

    MAPA_CLASSES = {
        # --- PERIGO ---
        'no-helmet': ('SEM CAPACETE', 1), 'no_helmet': ('SEM CAPACETE', 1),
        'no-goggles': ('SEM OCULOS', 1), 'no_goggles': ('SEM OCULOS', 1),
        'no-vest': ('SEM COLETE', 1), 'no_vest': ('SEM COLETE', 1),
        'no-glove': ('SEM LUVAS', 1), 'no_glove': ('SEM LUVAS', 1),
        'no-earplug': ('SEM PROTETOR', 1), 'no_earplug': ('SEM PROTETOR', 1),
        'no-hairnet': ('SEM TOUCA', 1), 'no_hairnet': ('SEM TOUCA', 1),
        
        # --- SEGURO ---
        'helmet': ('Capacete', 0), 'Helmet': ('Capacete', 0), '1-2-helmet': ('Capacete', 0),
        '3-4-helmet': ('Capacete', 0), 'Full-face-helmet': ('Capacete', 0),
        'goggles': ('Oculos', 0),
        'vest': ('Colete', 0),
        'glove': ('Luvas', 0),
        'earplug': ('Protetor', 0),
        'hairnet': ('Touca', 0),
        'boots': ('Botas', 0),
        
        # --- NEUTRO ---
        'person': ('Pessoa', 2),
        'cables': ('Fios/Cabos', 2)
    }

    LIMITES_CONFIANCA = {
        'Luvas': 0.15,      
        'Oculos': 0.10,     
        'Protetor': 0.15,   
        'Capacete': 0.30,   
        'Colete': 0.40,     
        'SEM OCULOS': 0.60, 
        'DEFAULT': 0.35     
    }

    def __init__(self, model_path="best.pt", resolution_ai=416):
        print(f"[INIT] Inicializando sistema...")
        self.lock = threading.Lock()
        self.running = False
        self.frame_atual = None
        self.memoria_objetos = [] 
        self.fps = 0
        self.resolution_ai = resolution_ai

        try:
            print(f"[INIT] Carregando YOLO ({model_path})...")
            self.model = YOLO(model_path)
            # Warmup
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy_img, verbose=False)
            print("[INIT] Sistema pronto!")
        except Exception as e:
            print(f"[FATAL] Erro ao carregar modelo: {e}")
            exit()

    def _get_conf_minima(self, nome_bonito):
        for chave, valor in self.LIMITES_CONFIANCA.items():
            if chave in nome_bonito:
                return valor
        return self.LIMITES_CONFIANCA['DEFAULT']

    def calcular_centro(self, box):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return cx, cy

    def atualizar_memoria(self, novas_deteccoes):
        agora = time.time()
        
        for novo in novas_deteccoes:
            novo_cx, novo_cy = self.calcular_centro(novo['box'])
            match_encontrado = False

            for antigo in self.memoria_objetos:
                if antigo['label'] == novo['label']:
                    antigo_cx, antigo_cy = self.calcular_centro(antigo['box'])
                    dist = math.hypot(novo_cx - antigo_cx, novo_cy - antigo_cy)
                    
                    # Como a imagem será redimensionada para 1920x1080 no final,
                    # precisamos ser tolerantes com a distância
                    if dist < self.DISTANCIA_MAX_MATCH:
                        antigo['box'] = novo['box']
                        antigo['time'] = agora
                        match_encontrado = True
                        break 
            
            if not match_encontrado:
                novo['time'] = agora
                self.memoria_objetos.append(novo)

        self.memoria_objetos = [
            obj for obj in self.memoria_objetos 
            if (agora - obj['time']) < self.TEMPO_MEMORIA
        ]

    def thread_ia(self):
        while self.running:
            if self.frame_atual is not None:
                try:
                    # Trabalha com a cópia original (pequena) para ser rápido
                    img = self.frame_atual.copy()
                    
                    results = self.model(img, imgsz=self.resolution_ai, conf=0.05, iou=0.6, verbose=False)
                    
                    deteccoes_brutas = []

                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            nome_raw = self.model.names[cls_id]

                            info = self.MAPA_CLASSES.get(nome_raw, (nome_raw, 2))
                            nome_bonito = info[0]
                            tipo_risco = info[1]

                            limit_min = self._get_conf_minima(nome_bonito)
                            if conf < limit_min: continue

                            if tipo_risco == 1: cor = self.COR_PERIGO
                            elif tipo_risco == 0: cor = self.COR_SEGURO
                            else: cor = self.COR_NEUTRO

                            deteccoes_brutas.append({
                                'box': (x1, y1, x2, y2),
                                'label': f"{nome_bonito}", 
                                'color': cor
                            })

                    with self.lock:
                        self.atualizar_memoria(deteccoes_brutas)
                except Exception as e:
                    print(f"[ERRO IA] {e}")

            time.sleep(0.015)

    def desenhar_visual(self, frame):
        if frame is None: return frame

        # --- AQUI ESTÁ A MÁGICA DO TAMANHO ---
        # Independente do tamanho da sua câmera, vamos forçar virar FULL HD
        # Isso garante que a janela fique cheia e a imagem também.
        frame_grande = cv2.resize(frame, (1920, 1080))
        
        # Precisamos recalcular a escala das caixas
        # Se a câmera era 640x480 e virou 1920x1080, as caixas têm que crescer
        h_orig, w_orig = frame.shape[:2]
        escala_x = 1920 / w_orig
        escala_y = 1080 / h_orig

        with self.lock:
            lista_desenho = list(self.memoria_objetos)

        for item in lista_desenho:
            # Pega coord original
            ox1, oy1, ox2, oy2 = item['box']
            
            # Aplica a escala para a tela grande
            x1 = int(ox1 * escala_x)
            y1 = int(oy1 * escala_y)
            x2 = int(ox2 * escala_x)
            y2 = int(oy2 * escala_y)
            
            label = item['label']
            color = item['color']
            
            cv2.rectangle(frame_grande, (x1, y1), (x2, y2), color, 3) 
            
            # Fonte maior porque a tela é Full HD
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(frame_grande, (x1, y1 - 40), (x1 + w, y1), color, -1)
            cv2.putText(frame_grande, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.putText(frame_grande, f"FPS: {int(self.fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        return frame_grande

    def iniciar(self):
        print("[INFO] Tentando abrir câmera (DirectShow)...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("[AVISO] DirectShow falhou. Tentando padrão...")
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("[FATAL] Nenhuma câmera encontrada.")
            return

        # Define uma resolução segura para capturar (640x480 é rápido e não trava)
        # Nós vamos esticar depois, então não precisa ser HD na entrada
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        ret, teste = cap.read()
        if not ret:
            print("[ERRO] Imagem da câmera vazia.")
            return
        
        self.running = True
        self.frame_atual = teste 

        t = threading.Thread(target=self.thread_ia)
        t.daemon = True
        t.start()

        print("--- SISTEMA ONLINE ---")
        print("Janela vai abrir em TELA CHEIA.")
        print("Pressione 'Q' para Sair")
        
        nome_janela = "Detector Tela Cheia"
        cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
        
        # --- COMANDO PARA FORÇAR TELA CHEIA (FULLSCREEN) ---
        cv2.setWindowProperty(nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        prev_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret: 
                continue

            self.frame_atual = frame
            
            # Aqui chamamos o método que desenha e já retorna a imagem em 1920x1080
            frame_final = self.desenhar_visual(frame)
            
            curr_time = time.time()
            diff = curr_time - prev_time
            if diff > 0:
                self.fps = 1 / diff
            prev_time = curr_time

            cv2.imshow(nome_janela, frame_final)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.running = False
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = DetectorEPIEstavel(model_path="best.pt", resolution_ai=416)
    app.iniciar()