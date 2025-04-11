import cv2
import numpy as np

def detectar_morangos(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # vermelho (vermelho pode estar em dois intervalos no HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # juntando as duas máscaras
    mask = cv2.bitwise_or(mask1, mask2)

    # Aplicando filtros para blurring
    mask = cv2.medianBlur(mask, 5)

    # Fazer contornos no que for vermelho
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    morango_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Ignora ruídos pequenos
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            morango_count += 1

    # Mostrar contagem
    cv2.putText(frame, f"Morangos detectados: {morango_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame

# Captura de vídeo (0 para webcam padrão)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_processado = detectar_morangos(frame)
    cv2.imshow("Detector de Morangos", frame_processado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()