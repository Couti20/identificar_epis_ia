import cv2
import threading
import time
from ultralytics import YOLO

class DetectorEPILocal:
    """
    Detector de EPIs usando YOLOv11 Local + Threading.
    O Threading garante que o vídeo não trave enquanto a IA pensa.
    """

    # --- CONFIGURAÇÕES DE CLASSES E CORES ---
    DANGER_CLASSES = {
        'no-helmet', 'no-glove', 'no-vest', 'no-goggles', 
        'no-earplug', 'no-hairnet', 'no-mask', 'no-boots',
        'sem-capacete', 'sem-luvas', 'sem-colete', 'sem-oculos',
        'sem-protetor', 'sem-touca'
    }
    
    SAFETY_CLASSES = {
        'helmet', 'glove', 'vest', 'goggles', 
        'earplug', 'hairnet', 'mask', 'boots',
        'capacete', 'luvas', 'colete', 'oculos'
    }

    COLOR_RED = (0, 0, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_YELLOW = (0, 255, 255) # Para coisas neutras (Pessoa, Fios)

    # Tradutor de nomes bagunçados para nomes bonitos
    EPI_ALIASES = {
        # Capacetes
        'Helmet': 'Capacete', 'helmet': 'Capacete', '1-2-helmet': 'Capacete',
        '3-4-helmet': 'Capacete', 'Full-face-helmet': 'Capacete',
        
        # Ausências (Perigo)
        'no-helmet': 'SEM CAPACETE', 'no_helmet': 'SEM CAPACETE',
        'no-goggles': 'SEM OCULOS', 'no_goggles': 'SEM OCULOS',
        'no-vest': 'SEM COLETE', 'no_vest': 'SEM COLETE',
        'no-glove': 'SEM LUVAS', 'no_glove': 'SEM LUVAS',
        'no-earplug': 'SEM PROTETOR', 'no_earplug': 'SEM PROTETOR',
        'no-hairnet': 'SEM TOUCA', 'no_hairnet': 'SEM TOUCA',
        
        # EPIs
        'goggles': 'Oculos',
        'vest': 'Colete',
        'glove': 'Luvas',
        'earplug': 'Protetor Auricular',
        'hairnet': 'Touca',
        'boots': 'Botas',
        
        # Outros
        'person': 'Pessoa',
        'cables': 'Fios/Cabos'
    }

    def __init__(self, model_path="best.pt", conf=0.25):
        print(f"[INIT] Carregando modelo YOLO local: {model_path}...")
        try:
            self.model = YOLO(model_path)
            print("[INIT] Modelo carregado com sucesso!")
        except Exception as e:
            print(f"[ERRO] Não foi possível carregar o modelo. Verifique se o arquivo existe.\nErro: {e}")
            exit()

        self.conf = conf
        self.running = False
        
        # Variáveis compartilhadas entre as Threads
        self.frame_atual = None          # O frame que a câmera acabou de pegar
        self.deteccoes_atuais = []       # As últimas caixas que a IA achou
        self.lock = threading.Lock()     # Segurança para as threads não brigarem

    def obter_cor_e_nome(self, nome_raw):
        """Define a cor e traduz o nome baseado na classe"""
        nome_raw_limpo = nome_raw.lower().strip()
        nome_bonito = self.EPI_ALIASES.get(nome_raw, nome_raw) # Tenta traduzir, senão usa o original

        # Lógica de cores
        if nome_raw_limpo in self.DANGER_CLASSES or "sem" in nome_bonito.lower() or "no_" in nome_raw_limpo:
            return self.COLOR_RED, nome_bonito
        elif nome_raw_limpo in self.SAFETY_CLASSES:
            return self.COLOR_GREEN, nome_bonito
        else:
            return self.COLOR_YELLOW, nome_bonito

    def cerebro_ia(self):
        """
        Esta função roda em SEGUNDO PLANO (outra thread).
        Ela pega o frame atual, passa no YOLO e atualiza as detecções.
        """
        while self.running:
            if self.frame_atual is not None:
                # Copia o frame para não travar a câmera
                img_para_ia = self.frame_atual.copy()

                # --- INFERÊNCIA YOLO ---
                # imgsz=320: Mantém rápido para CPU
                # conf=self.conf: Sua régua de qualidade
                # iou=0.6: Evita caixas duplicadas
                results = self.model(img_para_ia, verbose=False, imgsz=320, conf=self.conf, iou=0.6)

                novas_deteccoes = []
                
                # Processa os resultados
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # Extrai dados
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        nome_raw = self.model.names[cls_id]

                        # --- FILTROS EXTRAS (OPCIONAL) ---
                        # Aqui você pode ignorar o "Sem Oculos" se a confiança for baixa
                        nome_bonito = self.EPI_ALIASES.get(nome_raw, nome_raw)
                        if "SEM OCULOS" in nome_bonito and conf < 0.50:
                            continue # Ignora falsos positivos de óculos

                        novas_deteccoes.append({
                            'coords': (x1, y1, x2, y2),
                            'conf': conf,
                            'nome_raw': nome_raw
                        })

                # Atualiza a lista oficial de detecções
                with self.lock:
                    self.deteccoes_atuais = novas_deteccoes
            
            # Pequena pausa para não fritar o processador (opcional)
            time.sleep(0.01)

    def iniciar(self):
        """Inicia a câmera e o loop visual"""
        cap = cv2.VideoCapture(0)
        
        # Tenta resolução HD
        cap.set(3, 1280)
        cap.set(4, 720)

        if not cap.isOpened():
            print("[ERRO] Câmera não encontrada.")
            return

        self.running = True
        
        # Inicia a Thread da IA (Cérebro separado)
        thread_ia = threading.Thread(target=self.cerebro_ia)
        thread_ia.daemon = True
        thread_ia.start()

        print("[INFO] Sistema rodando! Pressione 'Q' para sair.")
        print("[INFO] Thread de vídeo: 30 FPS | Thread de IA: Rodando em paralelo.")

        # Janela maximizada
        cv2.namedWindow("Detector Profissional Local", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detector Profissional Local", 1280, 720)

        while True:
            ret, frame = cap.read()
            if not ret: break

            # Atualiza o frame para a IA ler
            self.frame_atual = frame

            # --- DESENHO ---
            # Pega as últimas detecções disponíveis (com segurança de Thread)
            with self.lock:
                deteccoes_para_desenhar = list(self.deteccoes_atuais)

            for det in deteccoes_para_desenhar:
                x1, y1, x2, y2 = det['coords']
                conf = det['conf']
                nome_raw = det['nome_raw']

                cor, texto = self.obter_cor_e_nome(nome_raw)
                
                # Desenha Retângulo
                cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                
                # Desenha Texto com Fundo
                label = f"{texto} {int(conf*100)}%"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), cor, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Informações na tela
            fps_ia = "IA: Rodando..." 
            cv2.putText(frame, "Modo: YOLOv11 Local (GPU/CPU)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            cv2.imshow("Detector Profissional Local", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.running = False
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Inicia o detector usando o best.pt da mesma pasta
    # conf=0.25: Ajuste a sensibilidade aqui
    app = DetectorEPILocal(model_path="best.pt", conf=0.25)
    app.iniciar()