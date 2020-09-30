import cv2
import numpy as np


def convert_rgb_to_bin(img):
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, thresh_img = cv2.threshold(gray_img, 140, 255, cv2.THRESH_BINARY)

	return thresh_img


def rotate_image(image, angle):
	image_center = tuple(np.array(image.shape[1::-1]) / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
	return result


def recognize_suits(img, cards, rotate=False):
	suits = ['clubs', 'diamonds', 'spades', 'hearts']
	cards_suits = dict()
	for i, card in enumerate(cards):
		x1, y1, x2, y2 = card
		crop_card = img[y1:y2, x1:x2]
		bin_card_img = convert_rgb_to_bin(crop_card)
		cards_suits[i] = None
		for suit in suits:
			template = cv2.imread(f'processing/suits_recognition/templates/{suit}.png', 0)
			angles = [0]
			# флаг указывающий на найденную масть
			is_current_suit = False
			# так как карты на руках наклонены, поиск происходит с различным наклоном шаблона
			if rotate:
				angles = [x - 45 for x in range(0, 100, 10)]

			for angle in angles:
				rot_template = rotate_image(template, angle)
				result = cv2.matchTemplate(bin_card_img, rot_template, cv2.TM_CCOEFF_NORMED)
				(_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)

				threshold = 0.9
				loc = np.where(result >= threshold)
				for _ in zip(*loc[::-1]):
					cards_suits[i] = suit
					is_current_suit = True
				if is_current_suit:
					break
			if is_current_suit:
				break
	return cards_suits