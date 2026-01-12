import cv2
import time
from threading import Thread
from ultralytics import YOLO

# ================= CONFIG =================

class Config:
    MODEL_PATH = "oitenta-tres.pt"
    CONF_GLOBAL = 0.40
    IMG_SIZE = 640
    DEVICE = "cpu"

    CAMERA_INDEX = 1
    WIDTH = 1280
    HEIGHT = 720

    IOU_LIMIT = 0.35

    ZOOM_ATIVADO = True
    ZOOM_FACTOR = 2.0
    ZOOM_MARGIN = 60

# ================= CLASSES =================

class Classes:
    PESSOA = 0
    OCULOS = 8
    LUVAS = 9
    BOTAS = 14
    CAPACETE = 10
    MAOS = 11
    CABECA = 12
    PE = 6
    COLETE = 16

    NOMES = {
        0: "Pessoa",
        8: "Óculos",
        9: "Luvas",
        10: "Capacete",
        11: "Mãos",
        12: "Cabeça",
        14: "Botas",
        16: "Colete",
        6: "Pé"
    }

    EPIS = [CAPACETE, OCULOS, LUVAS, BOTAS, COLETE]

# ================= CORES =================

class Cores:
    VERDE = (0,255,0)
    VERMELHO = (0,0,255)
    CIANO = (255,255,0)
    BRANCO = (255,255,255)
    PRETO = (0,0,0)
    AMARELO = (0,255,255)

# ================= ZOOM =================

ZOOM_THRESHOLDS = {
    Classes.OCULOS: (0.45, 0.80),
    Classes.LUVAS: (0.45, 0.78),
    Classes.BOTAS: (0.50, 0.80),
}

def precisa_zoom(box):
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    if cls not in ZOOM_THRESHOLDS:
        return False
    cmin, cmax = ZOOM_THRESHOLDS[cls]
    return cmin <= conf < cmax

def extrair_zoom(frame, box):
    h, w = frame.shape[:2]
    x1,y1,x2,y2 = map(int, box.xyxy[0])

    x1 = max(0, x1-Config.ZOOM_MARGIN)
    y1 = max(0, y1-Config.ZOOM_MARGIN)
    x2 = min(w, x2+Config.ZOOM_MARGIN)
    y2 = min(h, y2+Config.ZOOM_MARGIN)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    return cv2.resize(roi, None, fx=Config.ZOOM_FACTOR, fy=Config.ZOOM_FACTOR)

# ================= IOU =================

def iou(b1, b2):
    x1,y1,x2,y2 = map(int, b1.xyxy[0])
    x1b,y1b,x2b,y2b = map(int, b2.xyxy[0])

    xi1, yi1 = max(x1,x1b), max(y1,y1b)
    xi2, yi2 = min(x2,x2b), min(y2,y2b)

    inter = max(0,xi2-xi1)*max(0,yi2-yi1)
    a1 = (x2-x1)*(y2-y1)
    a2 = (x2b-x1b)*(y2b-y1b)
    return inter/(a1+a2-inter+1e-6)

# ================= MAIN =================

def main():
    model = YOLO(Config.MODEL_PATH)
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    cap.set(3, Config.WIDTH)
    cap.set(4, Config.HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=Config.IMG_SIZE, conf=Config.CONF_GLOBAL, device="cpu", verbose=False)
        boxes = results[0].boxes

        epis = []
        corpos = []
        confirmados_zoom = set()

        for b in boxes:
            cls = int(b.cls[0])
            if cls in Classes.EPIS:
                epis.append(b)
                if Config.ZOOM_ATIVADO and precisa_zoom(b):
                    roi = extrair_zoom(frame, b)
                    if roi is not None:
                        rz = model(roi, imgsz=640, conf=0.30, device="cpu", verbose=False)
                        for bb in rz[0].boxes:
                            if int(bb.cls[0]) == cls:
                                confirmados_zoom.add(b)
            else:
                corpos.append(b)

        riscos = 0
        pessoas = sum(1 for b in boxes if int(b.cls[0])==Classes.PESSOA)

        for c in corpos:
            ok = any(iou(c,e)>=Config.IOU_LIMIT for e in epis+list(confirmados_zoom))
            if not ok:
                riscos += 1
                x1,y1,x2,y2 = map(int,c.xyxy[0])
                cv2.rectangle(frame,(x1,y1),(x2,y2),Cores.VERMELHO,2)
                cv2.putText(frame,"SEM EPI",(x1,y1-5),0,0.6,Cores.VERMELHO,2)

        for e in epis:
            x1,y1,x2,y2 = map(int,e.xyxy[0])
            nome = Classes.NOMES[int(e.cls[0])]
            cor = Cores.VERDE
            if e in confirmados_zoom:
                nome += " (ZOOM)"
                cor = Cores.CIANO
            cv2.rectangle(frame,(x1,y1),(x2,y2),cor,2)
            cv2.putText(frame,nome,(x1,y1-5),0,0.6,cor,2)

        cv2.rectangle(frame,(0,0),(300,90),Cores.PRETO,-1)
        cv2.putText(frame,f"Pessoas: {pessoas}",(10,30),0,0.8,Cores.AMARELO,2)
        cv2.putText(frame,f"Riscos: {riscos}",(10,65),0,0.8,Cores.VERMELHO if riscos else Cores.VERDE,2)

        cv2.imshow("EPI AI", frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
