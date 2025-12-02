import cv2
import threading
import time
import numpy as np
from ultralytics import YOLO

class DetectorEPIPro:
    """
    Versão Final: Otimizada para Performance e Estabilidade.
    Inclui monitor de FPS, Warmup da IA e renderização visual aprimorada.
    """

    # --- CONFIGURAÇÕES VISUAIS ---
    # Cores em BGR (Blue, Green, Red)
    COR_PERIGO = (0, 0, 255)    # Vermelho
    COR_SEGURO = (0, 255, 0)    # Verde
    COR_NEUTRO = (255, 200, 0)  # Azul Claro/Ciano

    # --- DICIONÁRIO DE TRADUÇÃO E CATEGORIZACÃO ---
    # Mapeia o nome "cru" do modelo para (Nome Bonito, Tipo de Risco)
    # Tipo 0: Seguro (Verde) | Tipo 1: Perigo (Vermelho) | Tipo 2: Neutro (Azul)
    MAPA_CLASSES = {
        # --- PERIGO (Falta de EPI) ---
        'no-helmet': ('SEM CAPACETE', 1), 'no_helmet': ('SEM CAPACETE', 1),
        'no-goggles': ('SEM OCULOS', 1), 'no_goggles': ('SEM OCULOS', 1),
        'no-vest': ('SEM COLETE', 1), 'no_vest': ('SEM COLETE', 1),
        'no-glove': ('SEM LUVAS', 1), 'no_glove': ('SEM LUVAS', 1),
        'no-earplug': ('SEM PROTETOR', 1), 'no_earplug': ('SEM PROTETOR', 1),
        'no-hairnet': ('SEM TOUCA', 1), 'no_hairnet': ('SEM TOUCA', 1),
        
        # --- SEGURO (EPI Presente) ---
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

    # --- CONFIGURAÇÃO DE SENSIBILIDADE ---
    # Define a confiança mínima para cada tipo de objeto aparecer
    # AJUSTADO: Valores reduzidos para facilitar detecção de Capacete e Óculos
    LIMITES_CONFIANCA = {
        'Luvas': 0.15,      
        'Oculos': 0.10,     # Baixado para 10% (Pega qualquer coisa parecida com óculos)
        'Protetor': 0.15,   
        'Capacete': 0.30,   # Baixado para 30% (Era 50%, estava muito rígido)
        'Colete': 0.40,     # Baixado levemente
        'SEM OCULOS': 0.60, # Mantido alto para evitar falso positivo na parede
        'DEFAULT': 0.35     # Baixado o padrão geral
    }

    def __init__(self, model_path="best.pt", resolution_ai=416):
        print(f"[INIT] Inicializando sistema...")
        self.lock = threading.Lock()
        self.running = False
        self.frame_atual = None
        self.deteccoes = []
        self.fps = 0
        self.resolution_ai = resolution_ai # 320 (rápido) ou 416 (preciso)

        # Carregamento do Modelo
        try:
            print(f"[INIT] Carregando modelo YOLO ({model_path})...")
            self.model = YOLO(model_path)
            
            # WARMUP (Aquecimento)
            # Roda uma inferência falsa para carregar a GPU/CPU antes do vídeo abrir
            print("[INIT] Aquecendo motor de IA...")
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy_img, verbose=False)
            print("[INIT] Sistema pronto!")
            
        except Exception as e:
            print(f"[FATAL] Erro ao carregar modelo: {e}")
            exit()

    def _get_conf_minima(self, nome_bonito):
        """Retorna a confiança mínima para aquele objeto específico"""
        for chave, valor in self.LIMITES_CONFIANCA.items():
            if chave in nome_bonito:
                return valor
        return self.LIMITES_CONFIANCA['DEFAULT']

    def thread_ia(self):
        """O Cérebro: Processa imagens em loop sem travar o vídeo"""
        while self.running:
            if self.frame_atual is not None:
                # Copia o frame para não haver conflito de leitura/escrita
                img = self.frame_atual.copy()

                # INFERÊNCIA
                # stream=True e verbose=False deixam mais rápido
                # AJUSTE: conf=0.05 para garantir que a IA não descarte nada antes da nossa filtragem
                results = self.model(img, imgsz=self.resolution_ai, conf=0.05, iou=0.6, verbose=False)

                novas_deteccoes = []

                for r in results:
                    for box in r.boxes:
                        # Extração de dados
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        nome_raw = self.model.names[cls_id]

                        # Tradução e Lógica
                        info = self.MAPA_CLASSES.get(nome_raw, (nome_raw, 2))
                        nome_bonito = info[0]
                        tipo_risco = info[1]

                        # Filtro de Confiança Personalizado
                        limit_min = self._get_conf_minima(nome_bonito)
                        if conf < limit_min:
                            continue

                        # Definição de Cor
                        if tipo_risco == 1: cor = self.COR_PERIGO
                        elif tipo_risco == 0: cor = self.COR_SEGURO
                        else: cor = self.COR_NEUTRO

                        novas_deteccoes.append({
                            'box': (x1, y1, x2, y2),
                            'label': f"{nome_bonito} {int(conf*100)}%",
                            'color': cor
                        })

                # Atualiza a lista compartilhada com segurança
                with self.lock:
                    self.deteccoes = novas_deteccoes

            # Pausa tática para liberar CPU para a renderização do vídeo
            time.sleep(0.015)

    def desenhar_visual(self, frame):
        """Desenha as caixas e textos no frame"""
        # Pega cópia das detecções atuais
        with self.lock:
            lista_desenho = list(self.deteccoes)

        for item in lista_desenho:
            x1, y1, x2, y2 = item['box']
            label = item['label']
            color = item['color']

            # Desenha Retângulo
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Fundo do Texto (Efeito visual bonito)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Mostra FPS
        cv2.putText(frame, f"FPS: {int(self.fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

    def iniciar(self):
        """Loop Principal de Vídeo"""
        cap = cv2.VideoCapture(0)
        # Tenta HD
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("[ERRO] Câmera não encontrada.")
            return

        self.running = True
        
        # Inicia Thread IA
        t = threading.Thread(target=self.thread_ia)
        t.daemon = True
        t.start()

        print("--- SISTEMA RODANDO ---")
        print("Pressione 'Q' para Sair")
        print("Pressione 'R' para alternar resolução IA (320/416)")

        cv2.namedWindow("Detector EPI Pro", cv2.WINDOW_NORMAL)
        
        prev_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret: break

            # Atualiza frame para a IA
            self.frame_atual = frame

            # Desenha resultados
            frame_final = self.desenhar_visual(frame)
            
            # Cálculo de FPS
            curr_time = time.time()
            self.fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            cv2.imshow("Detector EPI Pro", frame_final)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Atalho para mudar resolução em tempo real
                if self.resolution_ai == 320:
                    self.resolution_ai = 416
                    print("[INFO] Modo Preciso (416px)")
                else:
                    self.resolution_ai = 320
                    print("[INFO] Modo Rápido (320px)")

        self.running = False
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Inicia o sistema
    # Use resolution_ai=320 se seu PC for lento
    # Use resolution_ai=416 se quiser detectar luvas melhor
    app = DetectorEPIPro(model_path="best.pt", resolution_ai=416)
    app.iniciar()