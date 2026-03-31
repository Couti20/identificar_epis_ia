import cv2
from ultralytics import YOLO

# -------------------
# Configurações
# -------------------
DEVICE = "cpu"  # rodando em CPU por enquanto
MODEL_PATH = "oitenta-tres.pt"  # caminho para o seu modelo treinado
CONFIDENCE_THRESHOLD = 0.5
DOUBT_THRESHOLD = 0.35
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Cores (BGR)
COLORS = {
    "luva": (0, 255, 0),            # verde
    "sem_luva": (0, 0, 255),        # vermelho
    "capacete": (255, 255, 0),      # ciano
    "sem_capacete": (0, 165, 255),  # laranja
    "mascara": (255, 0, 255),       # magenta
    "sem_mascara": (255, 0, 0),     # azul
    "duvida": (128, 128, 128)       # cinza
}

# -------------------
# Carregando modelo
# -------------------
model = YOLO(MODEL_PATH)

# -------------------
# Função para processar detecções
# -------------------
def process_detections(frame, results):
    people_count = 0
    non_compliance = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls].lower()

            # Contagem de pessoas
            if "person" in class_name or "pessoa" in class_name:
                people_count += 1

            # Cor e label
            if conf < DOUBT_THRESHOLD:
                color = COLORS["duvida"]
                label = f"{class_name}? ({conf:.2f})"
            else:
                if class_name == "luva":
                    color = COLORS["luva"]
                elif class_name == "sem_luva":
                    color = COLORS["sem_luva"]
                    non_compliance += 1
                elif class_name == "capacete":
                    color = COLORS["capacete"]
                elif class_name == "sem_capacete":
                    color = COLORS["sem_capacete"]
                    non_compliance += 1
                elif class_name == "mascara":
                    color = COLORS["mascara"]
                elif class_name == "sem_mascara":
                    color = COLORS["sem_mascara"]
                    non_compliance += 1
                else:
                    color = (255, 255, 255)  # default
                label = f"{class_name} ({conf:.2f})"

            # Desenhar retângulo e label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5), FONT, 0.5, color, 1)

    # Painel no canto esquerdo
    cv2.rectangle(frame, (0, 0), (250, 80), (50, 50, 50), -1)
    cv2.putText(frame, f"Pessoas: {people_count}", (10, 25), FONT, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Não conformes: {non_compliance}", (10, 55), FONT, 0.7, (0, 0, 255), 2)

    return frame

# -------------------
# Loop principal
# -------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao abrir webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, device=DEVICE, conf=CONFIDENCE_THRESHOLD)

        frame = process_detections(frame, results)

        cv2.imshow("Detecção de EPIs", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
