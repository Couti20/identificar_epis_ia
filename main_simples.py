"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     SISTEMA DE DETECÇÃO DE EPIs v2.0                         ║
║                    Monitoramento de Segurança em Tempo Real                  ║
║                          VERSÃO SIMPLES (SEM ZOOM)                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Autor: Sistema de IA para Segurança do Trabalho                             ║
║  Modelo: YOLOv8/v11 - Treinado com 83% de precisão                           ║
║  Resolução: Full HD (1920x1080)                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import time
from threading import Thread
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURAÇÕES
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    """Configurações centralizadas do sistema."""
    
    # Modelo
    MODELO_PATH = 'oitenta-tres.pt'
    CONFIANCA = 0.40              # Limiar mínimo de confiança (40%)
    TAMANHO_INFERENCIA = 640      # Resolução de entrada do YOLO
    
    # Câmera
    CAMERA_INDEX = 0              # 0 = Webcam integrada, 1 = USB externa
    CAMERA_LARGURA = 1920
    CAMERA_ALTURA = 1080
    
    # Lógica de Detecção
    IOU_LIMITE = 0.40             # IoU mínimo para considerar EPI em uso (40%)


# ══════════════════════════════════════════════════════════════════════════════
#                          MAPEAMENTO DE CLASSES
# ══════════════════════════════════════════════════════════════════════════════

class Classes:
    """Índices e nomes das classes do modelo."""
    
    # Índices das Classes
    PESSOA = 0
    ORELHA = 1
    PROTETOR_AURICULAR = 2
    ROSTO = 3
    PROTETOR_FACIAL = 4
    MASCARA = 5
    PE = 6
    FERRAMENTA = 7
    OCULOS = 8
    LUVAS = 9
    CAPACETE = 10
    MAOS = 11
    CABECA = 12
    TRAJE_MEDICO = 13
    BOTAS = 14
    TRAJE_SEGURANCA = 15
    COLETE = 16
    
    # Nomes para exibição
    NOMES = {
        0: 'Pessoa', 1: 'Orelha', 2: 'Protetor Auricular', 3: 'Rosto',
        4: 'Protetor Facial', 5: 'Máscara', 6: 'Pé', 7: 'Ferramenta',
        8: 'Óculos', 9: 'Luvas', 10: 'Capacete', 11: 'Mãos',
        12: 'Cabeça', 13: 'Traje Médico', 14: 'Botas',
        15: 'Traje Segurança', 16: 'Colete'
    }
    
    # EPIs de segurança (Verde)
    EPIS = [CAPACETE, COLETE, OCULOS, LUVAS, BOTAS]
    
    # Partes do corpo que indicam falta de EPI (Vermelho)
    CORPO = [CABECA, ROSTO, MAOS, PE]
    
    # Todos os itens monitorados
    MONITORADOS = EPIS + CORPO + [PESSOA]


# ══════════════════════════════════════════════════════════════════════════════
#                               CORES
# ══════════════════════════════════════════════════════════════════════════════

class Cores:
    """Paleta de cores BGR para OpenCV."""
    VERDE = (0, 255, 0)
    VERMELHO = (0, 0, 255)
    AMARELO = (0, 255, 255)
    PRETO = (0, 0, 0)
    BRANCO = (255, 255, 255)
    AZUL = (255, 100, 0)


# ══════════════════════════════════════════════════════════════════════════════
#                        STREAM DE WEBCAM (THREADING)
# ══════════════════════════════════════════════════════════════════════════════

class WebcamStream:
    """
    Captura de vídeo otimizada com threading.
    
    Benefícios:
    - Não bloqueia o processamento principal
    - Buffer mínimo para frame mais recente
    - Suporte a DirectShow no Windows
    """
    
    def __init__(self, src: int = 0, largura: int = 1920, altura: int = 1080):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.stopped = not self.stream.isOpened()
        self.frame = None
        
        if not self.stopped:
            _, self.frame = self.stream.read()
    
    def start(self) -> 'WebcamStream':
        """Inicia a thread de captura."""
        if not self.stopped:
            Thread(target=self._update, daemon=True).start()
        return self
    
    def _update(self) -> None:
        """Loop de captura contínua."""
        while not self.stopped:
            _, self.frame = self.stream.read()
    
    def read(self):
        """Retorna o frame mais recente."""
        return self.frame
    
    def stop(self) -> None:
        """Para a captura e libera recursos."""
        self.stopped = True
        self.stream.release()


# ══════════════════════════════════════════════════════════════════════════════
#                        PROCESSADOR DE DETECÇÕES
# ══════════════════════════════════════════════════════════════════════════════

class ProcessadorEPI:
    """
    Processa detecções YOLO e aplica lógica de conformidade de EPIs.
    
    Usa IoU (Intersection over Union) para determinar se um EPI está
    sendo usado corretamente sobre a parte do corpo correspondente.
    """
    
    # Regras de associação: parte_corpo -> epi_correspondente
    REGRAS = {
        Classes.CABECA: Classes.CAPACETE,
        Classes.ROSTO: Classes.OCULOS,
        Classes.MAOS: Classes.LUVAS,
        Classes.PE: Classes.BOTAS,
    }
    
    # Mensagens de não-conformidade
    ALERTAS = {
        Classes.CABECA: "SEM CAPACETE",
        Classes.ROSTO: "SEM ÓCULOS",
        Classes.MAOS: "SEM LUVAS",
        Classes.PE: "SEM BOTAS",
        Classes.PESSOA: "SEM COLETE",
    }
    
    @staticmethod
    def calcular_iou(box1, box2) -> float:
        """
        Calcula o IoU (Intersection over Union) entre duas caixas.
        
        Args:
            box1: Objeto box do YOLO (parte do corpo)
            box2: Objeto box do YOLO (EPI)
            
        Returns:
            float: Valor de IoU entre 0 e 1
        """
        # Coordenadas das caixas
        x1_1, y1_1, x2_1, y2_1 = map(int, box1.xyxy[0])
        x1_2, y1_2, x2_2, y2_2 = map(int, box2.xyxy[0])
        
        # Coordenadas da intersecção
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        
        # Área da intersecção
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        area_inter = inter_w * inter_h
        
        # Áreas das caixas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # IoU = Intersecção / União
        area_uniao = area1 + area2 - area_inter
        return area_inter / area_uniao if area_uniao > 0 else 0.0
    
    @staticmethod
    def tem_epi(box_corpo, lista_epis: list, iou_limite: float) -> bool:
        """Verifica se algum EPI cobre a parte do corpo com IoU suficiente."""
        for epi in lista_epis:
            if ProcessadorEPI.calcular_iou(box_corpo, epi) >= iou_limite:
                return True
        return False
    
    @staticmethod
    def processar(results, iou_limite: float) -> tuple:
        """
        Processa as detecções YOLO e retorna lista de anotações.
        
        Args:
            results: Resultado da inferência YOLO
            iou_limite: Limite mínimo de IoU para considerar EPI válido
            
        Returns:
            tuple: (anotações, qtd_pessoas)
        """
        boxes = results[0].boxes
        
        # Separa detecções por classe
        deteccoes = {cls: [] for cls in Classes.MONITORADOS}
        for box in boxes:
            cls = int(box.cls[0])
            if cls in deteccoes:
                deteccoes[cls].append(box)
        
        anotacoes = []
        
        # ────────────────────────────────────────────────────────────────
        # 1. COLETE: Verifica PESSOA contra COLETE
        # ────────────────────────────────────────────────────────────────
        for pessoa in deteccoes[Classes.PESSOA]:
            if not ProcessadorEPI.tem_epi(pessoa, deteccoes[Classes.COLETE], 0.01):
                anotacoes.append({
                    'box': pessoa,
                    'cor': Cores.VERMELHO,
                    'msg': ProcessadorEPI.ALERTAS[Classes.PESSOA],
                    'espessura': 2
                })
        
        # Adiciona coletes detectados (verde)
        for colete in deteccoes[Classes.COLETE]:
            anotacoes.append({
                'box': colete,
                'cor': Cores.VERDE,
                'msg': Classes.NOMES[Classes.COLETE],
                'espessura': 2
            })
        
        # ────────────────────────────────────────────────────────────────
        # 2. DEMAIS EPIs: Aplica regras de associação
        # ────────────────────────────────────────────────────────────────
        for parte_corpo, epi_classe in ProcessadorEPI.REGRAS.items():
            # Verifica cada parte do corpo detectada
            for box_corpo in deteccoes[parte_corpo]:
                if not ProcessadorEPI.tem_epi(box_corpo, deteccoes[epi_classe], iou_limite):
                    anotacoes.append({
                        'box': box_corpo,
                        'cor': Cores.VERMELHO,
                        'msg': ProcessadorEPI.ALERTAS[parte_corpo],
                        'espessura': 2
                    })
            
            # Adiciona EPIs detectados (verde)
            for epi in deteccoes[epi_classe]:
                anotacoes.append({
                    'box': epi,
                    'cor': Cores.VERDE,
                    'msg': Classes.NOMES[epi_classe],
                    'espessura': 2
                })
        
        return anotacoes, len(deteccoes[Classes.PESSOA])


# ══════════════════════════════════════════════════════════════════════════════
#                           RENDERIZADOR VISUAL
# ══════════════════════════════════════════════════════════════════════════════

class Renderizador:
    """Desenha as anotações e informações na tela."""
    
    @staticmethod
    def desenhar_caixa(frame, anotacao: dict) -> None:
        """Desenha uma caixa delimitadora com label."""
        box = anotacao['box']
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cor = anotacao['cor']
        msg = anotacao['msg']
        esp = anotacao['espessura']
        
        # Retângulo da detecção
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, esp)
        
        # Label com fundo
        (w, h), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + w + 10, y1), cor, -1)
        cv2.putText(frame, msg, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, Cores.BRANCO, 1)
    
    @staticmethod
    def desenhar_painel(frame, qtd_pessoas: int, qtd_riscos: int) -> None:
        """Desenha o painel informativo no canto superior."""
        # Fundo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (320, 100), Cores.PRETO, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Texto: Quantidade de pessoas
        cv2.putText(frame, f"PESSOAS: {qtd_pessoas}", (10, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, Cores.AMARELO, 1)
        
        # Texto: Status de segurança
        if qtd_riscos > 0:
            cv2.putText(frame, f"RISCO ATIVO: {qtd_riscos}", (10, 75),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, Cores.VERMELHO, 1)
        else:
            cv2.putText(frame, "SEGURANÇA OK", (10, 75),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, Cores.VERDE, 1)
    
    @staticmethod
    def desenhar_fps(frame, fps: int) -> None:
        """Desenha o contador de FPS no canto inferior direito."""
        h, w = frame.shape[:2]
        cv2.putText(frame, f"FPS: {fps}", (w - 150, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, Cores.BRANCO, 1)
    
    @staticmethod
    def renderizar(frame, anotacoes: list, qtd_pessoas: int, fps: int):
        """Renderiza todas as anotações no frame."""
        # Desenha todas as caixas
        for anotacao in anotacoes:
            Renderizador.desenhar_caixa(frame, anotacao)
        
        # Conta riscos
        qtd_riscos = sum(1 for a in anotacoes if a['cor'] == Cores.VERMELHO)
        
        # Desenha informações
        Renderizador.desenhar_painel(frame, qtd_pessoas, qtd_riscos)
        Renderizador.desenhar_fps(frame, fps)
        
        return frame


# ══════════════════════════════════════════════════════════════════════════════
#                            PROGRAMA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Função principal do sistema de detecção de EPIs."""
    
    print("=" * 60)
    print("   SISTEMA DE DETECÇÃO DE EPIs v2.0")
    print("   Monitoramento de Segurança em Tempo Real")
    print("   VERSÃO SIMPLES (SEM ZOOM)")
    print("=" * 60)
    
    # ──────────────────────────────────────────────────────────────
    # Carrega o modelo YOLO
    # ──────────────────────────────────────────────────────────────
    print(f"\n[1/3] Carregando modelo: {Config.MODELO_PATH}")
    try:
        modelo = YOLO(Config.MODELO_PATH)
        print("      ✓ Modelo carregado com sucesso!")
    except Exception as e:
        print(f"      ✗ Erro ao carregar modelo: {e}")
        return
    
    # ──────────────────────────────────────────────────────────────
    # Inicializa a câmera
    # ──────────────────────────────────────────────────────────────
    print(f"\n[2/3] Conectando câmera (índice {Config.CAMERA_INDEX})...")
    webcam = WebcamStream(
        src=Config.CAMERA_INDEX,
        largura=Config.CAMERA_LARGURA,
        altura=Config.CAMERA_ALTURA
    ).start()
    
    time.sleep(2.0)  # Aguarda estabilização
    
    if webcam.stopped:
        print("      ✗ Erro: Câmera não encontrada!")
        return
    
    print(f"      ✓ Câmera conectada ({Config.CAMERA_LARGURA}x{Config.CAMERA_ALTURA})")
    
    # ──────────────────────────────────────────────────────────────
    # Configura janela em tela cheia
    # ──────────────────────────────────────────────────────────────
    print("\n[3/3] Iniciando monitoramento...")
    NOME_JANELA = "Monitoramento EPI - Pressione 'Q' para sair"
    cv2.namedWindow(NOME_JANELA, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(NOME_JANELA, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    print("\n" + "=" * 60)
    print("   ✓ SISTEMA ATIVO - Pressione 'Q' para encerrar")
    print("=" * 60 + "\n")
    
    # ──────────────────────────────────────────────────────────────
    # Loop principal
    # ──────────────────────────────────────────────────────────────
    fps_inicio = time.time()
    fps = 0
    
    try:
        while True:
            frame = webcam.read()
            if frame is None:
                continue
            
            # Inferência YOLO
            results = modelo(
                frame.copy(),
                imgsz=Config.TAMANHO_INFERENCIA,
                conf=Config.CONFIANCA,
                verbose=False
            )
            
            # Processa detecções
            anotacoes, qtd_pessoas = ProcessadorEPI.processar(
                results, 
                Config.IOU_LIMITE
            )
            
            # Calcula FPS
            fps_fim = time.time()
            if fps_fim - fps_inicio > 0:
                fps = int(1 / (fps_fim - fps_inicio))
            fps_inicio = fps_fim
            
            # Renderiza e exibe
            frame_final = Renderizador.renderizar(
                frame, anotacoes, qtd_pessoas, fps
            )
            cv2.imshow(NOME_JANELA, frame_final)
            
            # Verifica tecla de saída
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n[!] Interrompido pelo usuário")
    
    finally:
        # ──────────────────────────────────────────────────────────
        # Limpeza
        # ──────────────────────────────────────────────────────────
        print("\n[*] Encerrando sistema...")
        webcam.stop()
        cv2.destroyAllWindows()
        print("[*] Sistema encerrado com sucesso!")


# ══════════════════════════════════════════════════════════════════════════════
#                              PONTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
