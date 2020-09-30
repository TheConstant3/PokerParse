from keras import models
from processing.suits_recognition.suit_recognition import convert_rgb_to_bin
import numpy as np
import cv2


class Recognizer:
    def __init__(self):
        self.model = models.load_model('processing/symbol_recognition/model')

    def predict_digit(self, img):
        h, w = img.shape
        # пропорциональное изменение размера с высотой 25 пикселей
        new_h = 25 if h > 25 else h
        new_w = w
        if new_h != h:
            new_w = int((new_h / h) * w)
        if new_w == 0 or new_h == 0:
            return None
        img = cv2.resize(img, (new_w - 3, new_h))

        # изображение помещается на черный фон
        # при измнении исходной картинки до размера 28*28 распознавание ухудшается
        black_bg = np.zeros((28, 28), np.uint8)
        black_bg[3:new_h + 3, 3:new_w] = img
        img = black_bg

        img = img / 255.0
        img = img.reshape(-1, 28, 28, 1)
        answer = self.model.predict(img)
        return answer


def recognize_symbols(img, objects, card=False):
    recognized_symbols = dict()

    recognizer = Recognizer()
    for i, coords in enumerate(objects):
        if coords == (0, 0, 0, 0):
            recognized_symbols[i] = None
            continue

        x1, y1, x2, y2 = coords
        symbol_img = img[y1:y2, x1:x2]
        # получение черного-белого изображения со значениями 0 и 255
        bw_symbol_img = convert_rgb_to_bin(symbol_img)

        # нейросеть была обучена на числах белого цвета на черном фоне
        # для корректного распознавания изображение куска белой карты инвентировалось
        if card:
            bw_symbol_img = (255-bw_symbol_img)

        contours, hierarchy = cv2.findContours(bw_symbol_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        symbols = dict()
        for idx, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            if hierarchy[0][idx][3] == -1 and h > w > 5:
                # ограничивающий прямоугольник находится близко к контуру числа
                # для корректного распознавания была добавлена прилегающая область
                stride = 2
                if x > stride and y > stride:
                    x -= stride
                    y -= stride
                    h += stride*2
                    w += stride*2
                image_for_recognize = bw_symbol_img[y:y + h, x:x + w]
                probs = recognizer.predict_digit(image_for_recognize)
                if probs is None:
                    continue
                value = str(np.argmax(probs))
                symbols[x] = value
        answer = ''
        queue = list(symbols.keys())
        # упорядочивание распознанных символов
        queue.sort()
        for x in queue:
            answer += symbols[x]

        A_J_K_Q = {'10': 'A', '11': 'J', '12': 'K', '13': 'Q'}
        if len(symbols.keys()) == 1 and answer in A_J_K_Q.keys():
            recognized_symbols[i] = A_J_K_Q[answer]
        else:
            recognized_symbols[i] = answer

    return recognized_symbols
