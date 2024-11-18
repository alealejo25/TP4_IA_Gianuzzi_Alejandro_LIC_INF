import cv2
import numpy as np
import matplotlib.pyplot as plt

# Función para cargar la imagen desde un archivo local
def load_image_from_file(file_path):
    return cv2.imread(file_path, cv2.IMREAD_COLOR)

# Ruta de la imagen local
image_path = "/home/ale/Documentos/2.jpg"  
# Cargar la imagen desde el archivo local
image = load_image_from_file(image_path)

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar suavizado para reducir ruido
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Detectar círculos usando la Transformada de Hough
circles = cv2.HoughCircles(
    blurred, 
    cv2.HOUGH_GRADIENT, 
    dp=1.2, 
    minDist=50, 
    param1=50, 
    param2=30, 
    minRadius=10, 
    maxRadius=100
)

# Dibujar los círculos detectados
if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        center = (circle[0], circle[1])  # Coordenadas del centro
        radius = circle[2]              # Radio del círculo
        cv2.circle(image, center, radius, (0, 255, 0), 2)  # Círculo
        cv2.circle(image, center, 3, (0, 0, 255), 3)       # Centro

# Mostrar resultados
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(gray, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Círculos Detectados")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
