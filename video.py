import cv2
import numpy as np

# ---------------------------
# 1. Captura de video
# ---------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo acceder a la cámara")
    exit()

# ---------------------------
# 2. Definición de rangos HSV
# ---------------------------

# Rojo (ajustado para evitar piel)
rojo_bajo1 = np.array([0, 150, 120])
rojo_alto1 = np.array([8, 255, 255])

rojo_bajo2 = np.array([172, 150, 120])
rojo_alto2 = np.array([180, 255, 255])


# Verde
verde_bajo = np.array([35, 80, 80])
verde_alto = np.array([85, 255, 255])

# Azul
azul_bajo = np.array([90, 80, 80])
azul_alto = np.array([130, 255, 255])

# Kernel para limpiar ruido
kernel = np.ones((5, 5), np.uint8)

# ---------------------------
# 3. Loop principal
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ---------------------------
    # 4. Máscaras por color
    # ---------------------------

    # Rojo
    mask_rojo1 = cv2.inRange(hsv, rojo_bajo1, rojo_alto1)
    mask_rojo2 = cv2.inRange(hsv, rojo_bajo2, rojo_alto2)
    mask_rojo = mask_rojo1 + mask_rojo2

    # Verde
    mask_verde = cv2.inRange(hsv, verde_bajo, verde_alto)

    # Azul
    mask_azul = cv2.inRange(hsv, azul_bajo, azul_alto)

    # Limpieza de ruido
    mask_rojo = cv2.morphologyEx(mask_rojo, cv2.MORPH_OPEN, kernel)
    mask_verde = cv2.morphologyEx(mask_verde, cv2.MORPH_OPEN, kernel)
    mask_azul = cv2.morphologyEx(mask_azul, cv2.MORPH_OPEN, kernel)

    # ---------------------------
    # 5. Función para detectar y dibujar
    # ---------------------------
    def detectar_color(mask, color_nombre, color_bgr):
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area > 1000:  # Filtrar ruido
                x, y, w, h = cv2.boundingRect(cnt)

                # Dibujar rectángulo
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)

                # Escribir texto
                cv2.putText(
                    frame,
                    color_nombre,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color_bgr,
                    2
                )

    # ---------------------------
    # 6. Detección de cada color
    # ---------------------------
    detectar_color(mask_rojo, "ROJO", (0, 0, 255))
    detectar_color(mask_verde, "VERDE", (0, 255, 0))
    detectar_color(mask_azul, "AZUL", (255, 0, 0))

    # Mostrar resultado
    cv2.imshow("Deteccion de Colores", frame)

    # Salir con ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ---------------------------
# 7. Liberar recursos
# ---------------------------
cap.release()
cv2.destroyAllWindows()