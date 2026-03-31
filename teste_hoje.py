import cv2
from ultralytics import YOLO

# ================= CONFIG =================

class Config:
    MODEL_PATH = "oitenta-tres.pt"
    DEVICE = "cpu"

    CONF_GLOBAL = 0.30
    CONF_CRITICO = 0.20
    IMG_SIZE = 640

    CAMERA_INDEX = 0
    WIDTH = 1280
    HEIGHT = 720

    IOU_PADRAO = 0.30
    IOU_CAPACETE = 0.18
    IOU_COLETE = 0.22

    ZOOM_ATIVADO = True
    ZOOM_FACTOR = 3.0
    ZOOM_MARGIN = 100

# ================= CLASSES =================

class Classes:
    PESSOA = 0
    PE = 6
    OCULOS = 8
    LUVAS = 9
    CAPACETE = 10
    MAOS = 11
    CABECA = 12
    BOTAS = 14
    COLETE = 16

    EPIS_CRITICOS = [CAPACETE, COLETE, BOTAS]
    EPIS = [CAPACETE, OCULOS, LUVAS, BOTAS, COLETE]

    NOMES = {
        CAPACETE: "Capacete",
        COLETE: "Colete",
        BOTAS: "Botas",
        LUVAS: "Luvas",
        OCULOS: "Óculos"
    }

# ================= MAPA EPI =================

EPI_POR_CORPO = {
    Classes.CABECA: Classes.CAPACETE,
    Classes.PESSOA: Classes.COLETE,
    Classes.PE: Classes.BOTAS,
    Classes.MAOS: Classes.LUVAS
}

NOME_FALTA = {
    Classes.CAPACETE: "Sem Capacete",
    Classes.COLETE: "Sem Colete",
    Classes.BOTAS: "Sem Botas",
    Classes.LUVAS: "Sem Luvas",
    Classes.OCULOS: "Sem Óculos"
}

# ================= CORES =================

class Cores:
    VERDE = (0,255,0)
    VERMELHO = (0,0,255)
    AMARELO = (0,255,255)
    CIANO = (255,255,0)
    PRETO = (0,0,0)

# ================= UTIL =================

def iou(b1, b2):
    x1,y1,x2,y2 = map(int, b1.xyxy[0])
    x1b,y1b,x2b,y2b = map(int, b2.xyxy[0])
    xi1, yi1 = max(x1,x1b), max(y1,y1b)
    xi2, yi2 = min(x2,x2b), min(y2,y2b)
    inter = max(0,xi2-xi1)*max(0,yi2-yi1)
    a1 = (x2-x1)*(y2-y1)
    a2 = (x2b-x1b)*(y2b-y1b)
    return inter/(a1+a2-inter+1e-6)

def extrair_zoom(frame, box):
    h,w = frame.shape[:2]
    x1,y1,x2,y2 = map(int, box.xyxy[0])
    x1 = max(0, x1-Config.ZOOM_MARGIN)
    y1 = max(0, y1-Config.ZOOM_MARGIN)
    x2 = min(w, x2+Config.ZOOM_MARGIN)
    y2 = min(h, y2+Config.ZOOM_MARGIN)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return cv2.resize(roi, None, fx=Config.ZOOM_FACTOR, fy=Config.ZOOM_FACTOR)

# ================= MAIN =================

def main():
    model = YOLO(Config.MODEL_PATH)
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    cap.set(3, Config.WIDTH)
    cap.set(4, Config.HEIGHT)

    # >>> TELA CHEIA <<<
    cv2.namedWindow("EPI AI - PRIORIDADE CRITICA", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "EPI AI - PRIORIDADE CRITICA",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=Config.IMG_SIZE,
                        conf=Config.CONF_GLOBAL,
                        device=Config.DEVICE,
                        verbose=False)

        boxes = results[0].boxes
        epis, corpos = [], []
        confirmados = set()

        pessoas = sum(1 for b in boxes if int(b.cls[0]) == Classes.PESSOA)

        for b in boxes:
            cls = int(b.cls[0])
            if cls in Classes.EPIS:
                epis.append(b)

                if Config.ZOOM_ATIVADO and cls in Classes.EPIS_CRITICOS:
                    roi = extrair_zoom(frame, b)
                    if roi is not None:
                        rz = model(roi, imgsz=640,
                                   conf=Config.CONF_CRITICO,
                                   device=Config.DEVICE,
                                   verbose=False)
                        for bb in rz[0].boxes:
                            if int(bb.cls[0]) == cls:
                                confirmados.add(b)
            else:
                corpos.append(b)

        riscos = 0

        for c in corpos:
            cls_corpo = int(c.cls[0])
            if cls_corpo not in EPI_POR_CORPO:
                continue

            epi = EPI_POR_CORPO[cls_corpo]

            limite = (
                Config.IOU_CAPACETE if epi == Classes.CAPACETE else
                Config.IOU_COLETE if epi == Classes.COLETE else
                Config.IOU_PADRAO
            )

            tem_epi = any(
                int(e.cls[0]) == epi and iou(c, e) >= limite
                for e in epis + list(confirmados)
            )

            if epi == Classes.CAPACETE:
                tem_epi = tem_epi or any(
                    int(e.cls[0]) == Classes.CAPACETE and iou(c, e) >= 0.12
                    for e in epis
                )

            if not tem_epi:
                riscos += 1
                x1,y1,x2,y2 = map(int, c.xyxy[0])
                cv2.rectangle(frame,(x1,y1),(x2,y2),Cores.VERMELHO,2)
                cv2.putText(frame, NOME_FALTA[epi],
                            (x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, Cores.VERMELHO, 2)

        for e in epis:
            x1,y1,x2,y2 = map(int,e.xyxy[0])
            cor = Cores.CIANO if e in confirmados else Cores.VERDE
            cv2.rectangle(frame,(x1,y1),(x2,y2),cor,2)
            cv2.putText(frame, Classes.NOMES[int(e.cls[0])],
                        (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, cor, 2)

        cv2.rectangle(frame,(0,0),(360,110),Cores.PRETO,-1)
        cv2.putText(frame,f"Pessoas: {pessoas}",(10,35),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,Cores.AMARELO,2)
        cv2.putText(frame,f"Riscos: {riscos}",(10,80),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,
                    Cores.VERMELHO if riscos else Cores.VERDE,2)

        cv2.imshow("EPI AI - PRIORIDADE CRITICA", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
