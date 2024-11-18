import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ruta de la imagen en tu PC
image_path = "/home/ale/Documentos/2.jpg"  

# Cargar la imagen desde la PC
image = cv2.imread(image_path)

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar detección de bordes (Canny)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Detectar líneas usando la Transformada de Hough
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)

# Dibujar las líneas detectadas sobre la imagen original
if lines is not None:
    for rho, theta in lines[:, 0]:
        # Convertir (ρ, θ) a coordenadas de dos puntos
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Mostrar resultados
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Bordes Detectados")
plt.imshow(edges, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Líneas Detectadas")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
