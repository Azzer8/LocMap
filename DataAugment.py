import cv2
import numpy as np
import random
import json

def rotate_point(point, angle, center):
    angle = np.deg2rad(angle)
    ox, oy = center
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    
    return [qx, qy]

def rotate_image_and_points(image, points, angle):
    """
    Поворачивает изображение и координаты меток на заданный угол.
    """
    (h, w) = image.shape[:3]
    center = (w / 2, h / 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    rotated_points = [rotate_point(point, angle, center) for point in points]
    
    return rotated_image, rotated_points

def scale_image_and_points(image, points, scale):
    """
    Масштабирует изображение и координаты меток с заданным коэффициентом.
    """
    h, w = image.shape[:2]
    scaled_image = cv2.resize(image, (int(w * scale), int(h * scale)))
    scaled_points = [[point[0] * scale, point[1] * scale] for point in points]
    
    return scaled_image, scaled_points

def random_crop_and_points(image, points, crop_size):
    """
    Случайно обрезает область изображения и соответствующим образом изменяет координаты меток.
    """
    h, w = image.shape[:2]
    ch, cw = crop_size
    
    if h < ch or w < cw:
        raise ValueError("Размер обрезки больше, чем размер изображения")
    
    x = random.randint(0, w - cw)
    y = random.randint(0, h - ch)
    
    cropped_image = image[y:y + ch, x:x + cw]
    cropped_points = [[point[0] - x, point[1] - y] for point in points if x <= point[0] <= x + cw and y <= point[1] <= y + ch]
    
    return cropped_image, cropped_points

def shift_image_and_points(image, points, shift_x, shift_y):
    """
    Сдвигает изображение и координаты меток на заданное расстояние по оси x и y.
    """
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    shifted_points = [[point[0] + shift_x, point[1] + shift_y] for point in points]
    
    return shifted_image, shifted_points

def load_labels(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    images_labels = []
    for line in lines:
        image_path, labels_json = line.strip().split('\t')
        labels = json.loads(labels_json)
        images_labels.append((image_path, labels))
    return images_labels

def save_labels(file_path, images_labels):
    with open(file_path, 'w', encoding='utf-8') as file:
        for image_path, labels in images_labels:
            labels_json = json.dumps(labels, ensure_ascii=False)
            file.write(f"{image_path}\t{labels_json}\n")