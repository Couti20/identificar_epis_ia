"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     SISTEMA DE DETECÇÃO DE EPIs v3.0                         ║
║                    Monitoramento de Segurança em Tempo Real                  ║
║                        COM ZOOM DE ATENÇÃO (AI Focus)                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Autor: Sistema de IA para Segurança do Trabalho                             ║
║  Modelo: YOLOv8/v11 - Treinado com 83% de precisão                           ║
║  Resolução: Full HD (1920x1080)                                              ║
║  Recurso: Zoom automático em áreas de dúvida para maior precisão             ║
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
    
    # ══════════════════════════════════════════════════════════════════════════
    # ZOOM DE ATENÇÃO (AI Focus) - Quando a IA tem dúvida, ela "aproxima" para ter certeza
    # ══════════════════════════════════════════════════════════════════════════
    ZOOM_ATIVADO = True           # Liga/desliga o zoom de atenção
    ZOOM_CONFIANCA_MINIMA = 0.50  # Abaixo disso, a IA está em "dúvida" e faz zoom
    ZOOM_CONFIANCA_MAXIMA = 0.75  # Acima disso, a IA tem "certeza" e não precisa zoom
    ZOOM_FATOR = 2.0              # Quanto ampliar a região (2x = dobro do tamanho)
    ZOOM_MARGEM = 50              # Pixels extras ao redor da região de interesse
    
    # EPIs que merecem zoom (objetos pequenos ou difíceis de detectar)
    ZOOM_CLASSES_PRIORITARIAS = [
        8,   # Óculos (pequeno, difícil de ver)
        9,   # Luvas (pode confundir com mãos)
        2,   # Protetor Auricular (muito pequeno)
        14,  # Botas (pode confundir com sapato comum)
    ]


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
    CIANO = (255, 255, 0)         # Para indicar zoom ativo
    LARANJA = (0, 165, 255)       # Para detecções com dúvida


# ══════════════════════════════════════════════════════════════════════════════
#                        ZOOM DE ATENÇÃO (AI FOCUS)
# ══════════════════════════════════════════════════════════════════════════════

class ZoomAtencao:
    """
    Sistema de Zoom de Atenção (AI Focus).
    
    Quando a IA detecta algo com confiança baixa (está em "dúvida"),
    ela recorta e amplia a região de interesse e reanalisa com mais detalhes.
    
    Isso é especialmente útil para:
    - Óculos (objeto pequeno)
    - Luvas (pode confundir com mãos)
    - Protetor auricular (muito pequeno)
    - Botas (pode confundir com sapato comum)
    """
    
    @staticmethod
    def extrair_regiao(frame, box, margem: int = 50, fator_zoom: float = 2.0):
        """
        Extrai e amplia uma região de interesse do frame.
        
        Args:
            frame: Imagem completa
            box: Objeto box do YOLO com coordenadas
            margem: Pixels extras ao redor
            fator_zoom: Fator de ampliação
            
        Returns:
            tuple: (regiao_ampliada, coordenadas_originais)
        """
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Adiciona margem e garante limites
        x1_m = max(0, x1 - margem)
        y1_m = max(0, y1 - margem)
        x2_m = min(w_frame, x2 + margem)
        y2_m = min(h_frame, y2 + margem)
        
        # Recorta a região
        regiao = frame[y1_m:y2_m, x1_m:x2_m]
        
        if regiao.size == 0:
            return None, None
        
        # Amplia a região (zoom digital)
        nova_largura = int(regiao.shape[1] * fator_zoom)
        nova_altura = int(regiao.shape[0] * fator_zoom)
        
        # Garante tamanho mínimo para o YOLO processar
        nova_largura = max(nova_largura, 320)
        nova_altura = max(nova_altura, 320)
        
        regiao_ampliada = cv2.resize(regiao, (nova_largura, nova_altura), 
                                      interpolation=cv2.INTER_CUBIC)
        
        return regiao_ampliada, (x1_m, y1_m, x2_m, y2_m)
    
    @staticmethod
    def detectar_com_zoom(modelo, frame, box_original, classe_alvo: int,
                          margem: int = 50, fator_zoom: float = 2.0,
                          confianca: float = 0.30) -> dict:
        """
        Reanalisa uma região específica com zoom para maior precisão.
        
        Args:
            modelo: Modelo YOLO carregado
            frame: Frame completo
            box_original: Box da detecção original (com dúvida)
            classe_alvo: Classe que estamos procurando
            margem: Margem ao redor da região
            fator_zoom: Fator de ampliação
            confianca: Confiança mínima para a segunda análise
            
        Returns:
            dict: Resultado do zoom {'encontrado': bool, 'confianca': float, 'box': ...}
        """
        regiao, coords = ZoomAtencao.extrair_regiao(frame, box_original, margem, fator_zoom)
        
        if regiao is None:
            return {'encontrado': False, 'confianca': 0.0, 'zoom_usado': False}
        
        # Roda inferência na região ampliada
        results = modelo(regiao, imgsz=640, conf=confianca, verbose=False)
        
        # Procura pela classe alvo na região ampliada
        melhor_confianca = 0.0
        encontrado = False
        
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls == classe_alvo and conf > melhor_confianca:
                melhor_confianca = conf
                encontrado = True
        
        return {
            'encontrado': encontrado,
            'confianca': melhor_confianca,
            'zoom_usado': True,
            'coords_zoom': coords
        }
    
    @staticmethod
    def precisa_zoom(box, classes_prioritarias: list, 
                     conf_minima: float, conf_maxima: float) -> bool:
        """
        Determina se uma detecção precisa de zoom para confirmação.
        
        Args:
            box: Objeto box do YOLO
            classes_prioritarias: Lista de classes que merecem zoom
            conf_minima: Confiança mínima (abaixo = muito incerto, ignora)
            conf_maxima: Confiança máxima (acima = certo, não precisa zoom)
            
        Returns:
            bool: True se precisa fazer zoom
        """
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Só faz zoom em classes prioritárias
        if cls not in classes_prioritarias:
            return False
        
        # Faixa de "dúvida": entre mínima e máxima
        return conf_minima <= conf < conf_maxima


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
    def processar(results, iou_limite: float, frame=None, modelo=None) -> tuple:
        """
        Processa as detecções YOLO e retorna lista de anotações.
        
        Args:
            results: Resultado da inferência YOLO
            iou_limite: Limite mínimo de IoU para considerar EPI válido
            frame: Frame original (necessário para zoom)
            modelo: Modelo YOLO (necessário para zoom)
            
        Returns:
            tuple: (anotações, qtd_pessoas, estatísticas_zoom)
        """
        boxes = results[0].boxes
        
        # Separa detecções por classe (agora com info de confiança)
        deteccoes = {cls: [] for cls in Classes.MONITORADOS}
        deteccoes_com_duvida = []  # Detecções que precisam de zoom
        
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls in deteccoes:
                deteccoes[cls].append(box)
                
                # Verifica se precisa de zoom (está em dúvida)
                if Config.ZOOM_ATIVADO and frame is not None and modelo is not None:
                    if ZoomAtencao.precisa_zoom(box, Config.ZOOM_CLASSES_PRIORITARIAS,
                                                 Config.ZOOM_CONFIANCA_MINIMA,
                                                 Config.ZOOM_CONFIANCA_MAXIMA):
                        deteccoes_com_duvida.append({
                            'box': box,
                            'classe': cls,
                            'confianca_original': conf
                        })
        
        # Estatísticas de zoom
        stats_zoom = {
            'total_duvidas': len(deteccoes_com_duvida),
            'zooms_realizados': 0,
            'confirmados': 0,
            'regioes_zoom': []
        }
        
        # ════════════════════════════════════════════════════════════════
        # ZOOM DE ATENÇÃO: Reanalisa detecções com dúvida
        # ════════════════════════════════════════════════════════════════
        epis_confirmados_zoom = {cls: [] for cls in Classes.EPIS}
        
        for item in deteccoes_com_duvida:
            box = item['box']
            cls = item['classe']
            
            # Faz zoom e reanalisa
            resultado_zoom = ZoomAtencao.detectar_com_zoom(
                modelo, frame, box, cls,
                margem=Config.ZOOM_MARGEM,
                fator_zoom=Config.ZOOM_FATOR,
                confianca=0.30  # Confiança mais baixa no zoom (já está ampliado)
            )
            
            stats_zoom['zooms_realizados'] += 1
            
            if resultado_zoom['encontrado']:
                stats_zoom['confirmados'] += 1
                # Guarda a região onde foi confirmado (para desenhar indicador)
                if resultado_zoom.get('coords_zoom'):
                    stats_zoom['regioes_zoom'].append({
                        'coords': resultado_zoom['coords_zoom'],
                        'classe': cls,
                        'confianca_zoom': resultado_zoom['confianca']
                    })
                # Marca como confirmado por zoom
                epis_confirmados_zoom[cls].append(box)
        
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
                    'espessura': 2,
                    'zoom': False
                })
        
        # Adiciona coletes detectados (verde)
        for colete in deteccoes[Classes.COLETE]:
            anotacoes.append({
                'box': colete,
                'cor': Cores.VERDE,
                'msg': Classes.NOMES[Classes.COLETE],
                'espessura': 2,
                'zoom': False
            })
        
        # ────────────────────────────────────────────────────────────────
        # 2. DEMAIS EPIs: Aplica regras de associação (COM ZOOM)
        # ────────────────────────────────────────────────────────────────
        for parte_corpo, epi_classe in ProcessadorEPI.REGRAS.items():
            # Combina EPIs detectados normalmente + confirmados por zoom
            epis_disponiveis = deteccoes[epi_classe] + epis_confirmados_zoom.get(epi_classe, [])
            
            # Verifica cada parte do corpo detectada
            for box_corpo in deteccoes[parte_corpo]:
                if not ProcessadorEPI.tem_epi(box_corpo, epis_disponiveis, iou_limite):
                    anotacoes.append({
                        'box': box_corpo,
                        'cor': Cores.VERMELHO,
                        'msg': ProcessadorEPI.ALERTAS[parte_corpo],
                        'espessura': 2,
                        'zoom': False
                    })
            
            # Adiciona EPIs detectados (verde) - marca se foi confirmado por zoom
            for epi in deteccoes[epi_classe]:
                conf = float(epi.conf[0])
                foi_zoom = epi in [item['box'] for item in deteccoes_com_duvida]
                confirmado_zoom = epi in epis_confirmados_zoom.get(epi_classe, [])
                
                # Define cor e mensagem baseado no status
                if confirmado_zoom:
                    cor = Cores.CIANO  # Ciano = confirmado por zoom
                    msg = f"{Classes.NOMES[epi_classe]} ✓ZOOM"
                elif conf >= Config.ZOOM_CONFIANCA_MAXIMA:
                    cor = Cores.VERDE  # Verde = certeza alta
                    msg = Classes.NOMES[epi_classe]
                else:
                    cor = Cores.VERDE
                    msg = f"{Classes.NOMES[epi_classe]} ({int(conf*100)}%)"
                
                anotacoes.append({
                    'box': epi,
                    'cor': cor,
                    'msg': msg,
                    'espessura': 3 if confirmado_zoom else 2,
                    'zoom': confirmado_zoom
                })
        
        return anotacoes, len(deteccoes[Classes.PESSOA]), stats_zoom


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
    def desenhar_painel(frame, qtd_pessoas: int, qtd_riscos: int, 
                        stats_zoom: dict = None) -> None:
        """Desenha o painel informativo no canto superior."""
        # Calcula altura do painel baseado nas informações
        altura_painel = 140 if Config.ZOOM_ATIVADO else 100
        
        # Fundo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (350, altura_painel), Cores.PRETO, -1)
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
        
        # Informações de Zoom (se ativado)
        if Config.ZOOM_ATIVADO and stats_zoom:
            zoom_info = f"ZOOM: {stats_zoom['confirmados']}/{stats_zoom['total_duvidas']} confirmados"
            cor_zoom = Cores.CIANO if stats_zoom['confirmados'] > 0 else Cores.BRANCO
            cv2.putText(frame, zoom_info, (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_zoom, 1)
    
    @staticmethod
    def desenhar_regioes_zoom(frame, stats_zoom: dict) -> None:
        """Desenha indicadores nas regiões onde o zoom foi usado."""
        if not stats_zoom or not stats_zoom.get('regioes_zoom'):
            return
        
        for regiao in stats_zoom['regioes_zoom']:
            x1, y1, x2, y2 = regiao['coords']
            # Desenha retângulo tracejado (cantos) para indicar região de zoom
            tamanho_canto = 20
            
            # Cantos superiores
            cv2.line(frame, (x1, y1), (x1 + tamanho_canto, y1), Cores.CIANO, 2)
            cv2.line(frame, (x1, y1), (x1, y1 + tamanho_canto), Cores.CIANO, 2)
            cv2.line(frame, (x2, y1), (x2 - tamanho_canto, y1), Cores.CIANO, 2)
            cv2.line(frame, (x2, y1), (x2, y1 + tamanho_canto), Cores.CIANO, 2)
            
            # Cantos inferiores
            cv2.line(frame, (x1, y2), (x1 + tamanho_canto, y2), Cores.CIANO, 2)
            cv2.line(frame, (x1, y2), (x1, y2 - tamanho_canto), Cores.CIANO, 2)
            cv2.line(frame, (x2, y2), (x2 - tamanho_canto, y2), Cores.CIANO, 2)
            cv2.line(frame, (x2, y2), (x2, y2 - tamanho_canto), Cores.CIANO, 2)
            
            # Ícone de lupa/zoom
            cv2.putText(frame, "🔍", (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, Cores.CIANO, 1)
    
    @staticmethod
    def desenhar_fps(frame, fps: int) -> None:
        """Desenha o contador de FPS no canto inferior direito."""
        h, w = frame.shape[:2]
        cv2.putText(frame, f"FPS: {fps}", (w - 150, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, Cores.BRANCO, 1)
    
    @staticmethod
    def renderizar(frame, anotacoes: list, qtd_pessoas: int, fps: int, 
                   stats_zoom: dict = None):
        """Renderiza todas as anotações no frame."""
        # Desenha indicadores de região de zoom primeiro (fica por baixo)
        if stats_zoom:
            Renderizador.desenhar_regioes_zoom(frame, stats_zoom)
        
        # Desenha todas as caixas
        for anotacao in anotacoes:
            Renderizador.desenhar_caixa(frame, anotacao)
        
        # Conta riscos
        qtd_riscos = sum(1 for a in anotacoes if a['cor'] == Cores.VERMELHO)
        
        # Desenha informações
        Renderizador.desenhar_painel(frame, qtd_pessoas, qtd_riscos, stats_zoom)
        Renderizador.desenhar_fps(frame, fps)
        
        return frame


# ══════════════════════════════════════════════════════════════════════════════
#                            PROGRAMA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Função principal do sistema de detecção de EPIs."""
    
    print("=" * 70)
    print("   SISTEMA DE DETECÇÃO DE EPIs v3.0")
    print("   Monitoramento de Segurança em Tempo Real")
    print("   COM ZOOM DE ATENÇÃO (AI Focus)")
    print("=" * 70)
    
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
    
    print("\n" + "=" * 70)
    print("   ✓ SISTEMA ATIVO - Pressione 'Q' para encerrar")
    if Config.ZOOM_ATIVADO:
        print("   🔍 ZOOM DE ATENÇÃO: ATIVADO")
        print(f"      Classes prioritárias: {Config.ZOOM_CLASSES_PRIORITARIAS}")
        print(f"      Faixa de dúvida: {int(Config.ZOOM_CONFIANCA_MINIMA*100)}% - {int(Config.ZOOM_CONFIANCA_MAXIMA*100)}%")
    print("=" * 70 + "\n")
    
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
            
            # Guarda cópia para zoom (evita conflito de leitura)
            frame_original = frame.copy()
            
            # Inferência YOLO principal
            results = modelo(
                frame_original,
                imgsz=Config.TAMANHO_INFERENCIA,
                conf=Config.CONFIANCA,
                verbose=False
            )
            
            # Processa detecções (COM ZOOM se ativado)
            anotacoes, qtd_pessoas, stats_zoom = ProcessadorEPI.processar(
                results, 
                Config.IOU_LIMITE,
                frame=frame_original if Config.ZOOM_ATIVADO else None,
                modelo=modelo if Config.ZOOM_ATIVADO else None
            )
            
            # Calcula FPS
            fps_fim = time.time()
            if fps_fim - fps_inicio > 0:
                fps = int(1 / (fps_fim - fps_inicio))
            fps_inicio = fps_fim
            
            # Renderiza e exibe
            frame_final = Renderizador.renderizar(
                frame, anotacoes, qtd_pessoas, fps, stats_zoom
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
