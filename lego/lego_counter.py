"""
lego_counter.py

Contador sencillo de piezas LEGO por color usando OpenCV.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Imagen de entrada")
    parser.add_argument("--out", default="lego/outputs", help="Carpeta de salida")
    args = parser.parse_args()

    img_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Leer imagen
    img = cv2.imread(str(img_path))
    if img is None:
        print("No se pudo leer la imagen.")
        return

    # Convertir a HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Definir rangos HSV para colores básicos
    colores = {
        "rojo": [(np.array([0, 100, 70]), np.array([10, 255, 255]))],
        "azul": [(np.array([95, 100, 70]), np.array([130, 255, 255]))],
        "verde": [(np.array([35, 80, 60]), np.array([85, 255, 255]))],
        "amarillo": [(np.array([20, 120, 70]), np.array([35, 255, 255]))],
        "morado": [(np.array([130, 100, 70]), np.array([160, 255, 255]))],
        "naranja": [(np.array([10, 100, 70]), np.array([20, 255, 255]))]
    }

    conteo = {}

    for nombre, rangos in colores.items():
        total_mask = np.zeros(hsv.shape[:2], dtype="uint8")
        for low, high in rangos:
            mask = cv2.inRange(hsv, low, high)
            total_mask = cv2.bitwise_or(total_mask, mask)

        # Limpiar ruido
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_clean = cv2.morphologyEx(total_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Buscar contornos
        contornos, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrar piezas pequeñas
        piezas = [c for c in contornos if cv2.contourArea(c) > 200]
        conteo[nombre] = len(piezas)

        # Dibujar
        for c in piezas:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.putText(img, nombre, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    print("Conteo:", conteo)

    # Guardar imagen con resultados
    out_path = out_dir / f"{img_path.stem}_resultado.png"
    cv2.imwrite(str(out_path), img)
    print("Guardado en:", out_path)


if __name__ == "__main__":
    main()
