import cv2
from ultralytics import YOLO

# --- CONFIGURAÇÃO INICIAL ---
print("Carregando modelo... Aguarde um momento.")
model = YOLO("best.pt")

# Dicionário para limpar os nomes (Traduz e simplifica)
nomes_bonitos = {
    # Capacetes
    'Helmet': 'Capacete', 'helmet': 'Capacete',
    '1-2-helmet': 'Capacete', '3-4-helmet': 'Capacete',
    'Full-face-helmet': 'Capacete',
    
    # Ausências (Perigo)
    'no-helmet': 'SEM CAPACETE!',
    'no-goggles': 'SEM OCULOS!',
    'no-vest': 'SEM COLETE!',
    'no-glove': 'SEM LUVAS!',
    'no_earplug': 'SEM PROTETOR!', # Ajustei conforme seu print
    'no_hairnet': 'SEM TOUCA!',    # Ajustei conforme seu print
    
    # EPIs
    'goggles': 'Oculos',
    'vest': 'Colete',
    'boots': 'Botas',
    'glove': 'Luvas',
    'earplug': 'Protetor Auricular',
    
    # Outros
    'cables': 'Fios Soltos',
    'person': 'Pessoa'
}

# Cores (B, G, R)
cor_verde = (0, 255, 0)      # Seguro
cor_vermelha = (0, 0, 255)   # Perigo
cor_neutra = (255, 0, 0)     # Neutro

# --- CONFIGURAÇÃO DA CÂMERA E TELA ---
cap = cv2.VideoCapture(0)

# Define a resolução de CAPTURA (Padrão 640x480 é mais rápido para processar)
cap.set(3, 640)
cap.set(4, 480)

# Cria uma janela que permite redimensionar
nome_janela = "Detector de EPIs (YOLOv11)"
cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)

# Força a janela a abrir GRANDE (1280x720) na sua tela
cv2.resizeWindow(nome_janela, 1280, 720)

print("✅ Câmera iniciada! A janela deve abrir grande agora.")
print("Pressione 'Q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao ler a câmera.")
        break

    # --- O PULO DO GATO (AJUSTE FINO) ---
    # conf=0.45: Só mostra se tiver 45% de certeza (Elimina quadrados gigantes na parede)
    # iou=0.5: Evita desenhar dois quadrados para o mesmo objeto
    # imgsz=320: Mantém a velocidade alta no seu notebook
    results = model(frame, stream=True, conf=0.45, iou=0.5, imgsz=320, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 1. Coordenadas
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 2. Identificação
            cls_id = int(box.cls[0])
            current_name = model.names[cls_id]
            label = nomes_bonitos.get(current_name, current_name)

            # 3. Definição de Cores
            if "SEM" in label:
                color = cor_vermelha
            elif label in ['Pessoa', 'Fios Soltos', 'Botas']:
                color = cor_neutra
            else:
                color = cor_verde

            # 4. Desenho (Retângulo mais grosso para ver melhor na tela grande)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Etiqueta
            # Calcula o tamanho do texto para fazer o fundo colorido certinho
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Mostra a imagem na janela (que já definimos como grande)
    cv2.imshow(nome_janela, frame)

    # Sai com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()