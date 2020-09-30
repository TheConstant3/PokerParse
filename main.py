from processing.symbol_recognition.cnn_recognizer import recognize_symbols
from processing.suits_recognition.suit_recognition import *
import argparse
import json


def get_balances(img):
	print('recognizing balances...')
	balances = [(360, 1400, 510, 1430),
				(20, 1190, 170, 1220),
				(20, 920, 170, 950),
				(20, 525, 170, 555),
				(162, 275, 312, 305),
				(555, 275, 705, 305),
				(700, 525, 850, 555),
				(700, 920, 850, 950),
				(700, 1190, 850, 1220)]

	return recognize_symbols(img, balances)


def get_bets(img):
	print('recognizing bets...')
	bets = [(0, 0, 0, 0),
			(177, 1165, 300, 1195),
			(177, 894, 300, 924),
			(0, 0, 0, 0),
			(0, 0, 0, 0),
			(0, 0, 0, 0),
			(0, 0, 0, 0),
			(0, 0, 0, 0),
			(0, 0, 0, 0)]

	return recognize_symbols(img, bets)


def get_pot(img):
	print('recognizing pot...')
	pots = [(370, 567, 500, 597)]
	return recognize_symbols(img, pots)


def get_table_cards(img):
	print('recognizing table cards...')
	table_suits_coords = [(155, 605, 205, 685),
						  (270, 605, 320, 685),
						  (385, 605, 435, 685),
						  (500, 605, 550, 685)]
	table_suits = recognize_suits(img, table_suits_coords)

	table_rank_coords = [(160, 605, 200, 650),
						 (275, 605, 315, 650),
						 (390, 605, 430, 650),
						 (505, 605, 545, 650)]
	table_ranks = recognize_symbols(img, table_rank_coords, card=True)

	return table_suits, table_ranks


def get_hand_cards(img):
	print('recognizing hand cards...')
	hand_suits_coords = [(520, 1290, 560, 1350),
						 (590, 1285, 630, 1345)]
	hand_suits = recognize_suits(img, hand_suits_coords, rotate=True)

	hand_rank_coords = [(525, 1293, 560, 1330),
						(600, 1285, 632, 1325)]
	hand_ranks = recognize_symbols(img, hand_rank_coords, card=True)

	return hand_suits, hand_ranks


def print_all_bboxes(img):
	# balances
	img = cv2.rectangle(img, (360, 1400), (510, 1430), (0, 255, 0), 2)  # me
	img = cv2.rectangle(img, (20, 1190), (170, 1220), (0, 255, 0), 2)
	img = cv2.rectangle(img, (20, 920), (170, 950), (0, 255, 0), 2)
	img = cv2.rectangle(img, (20, 525), (170, 555), (0, 255, 0), 2)
	img = cv2.rectangle(img, (162, 275), (312, 305), (0, 255, 0), 2)
	img = cv2.rectangle(img, (555, 275), (705, 305), (0, 255, 0), 2)
	img = cv2.rectangle(img, (700, 525), (850, 555), (0, 255, 0), 2)
	img = cv2.rectangle(img, (700, 920), (850, 950), (0, 255, 0), 2)
	img = cv2.rectangle(img, (700, 1190), (850, 1220), (0, 255, 0), 2)

	# bets
	img = cv2.rectangle(img, (177, 1165), (300, 1195), (0, 0, 255), 2)
	img = cv2.rectangle(img, (177, 894), (300, 924), (0, 0, 255), 2)

	# pot
	img = cv2.rectangle(img, (370, 567), (500, 597), (255, 0, 0), 2)

	# table suits
	img = cv2.rectangle(img, (155, 605), (205, 685), (0, 255, 0), 2)
	img = cv2.rectangle(img, (270, 605), (320, 685), (0, 255, 0), 2)
	img = cv2.rectangle(img, (385, 605), (435, 685), (0, 255, 0), 2)
	img = cv2.rectangle(img, (500, 605), (550, 685), (0, 255, 0), 2)

	# table ranks
	img = cv2.rectangle(img, (160, 605), (200, 650), (0, 255, 0), 2)
	img = cv2.rectangle(img, (275, 605), (315, 650), (0, 255, 0), 2)
	img = cv2.rectangle(img, (390, 605), (430, 650), (0, 255, 0), 2)
	img = cv2.rectangle(img, (505, 605), (545, 650), (0, 255, 0), 2)

	# hand suits
	img = cv2.rectangle(img, (520, 1290), (560, 1350), (255, 255, 0), 2)
	img = cv2.rectangle(img, (590, 1285), (630, 1345), (255, 255, 0), 2)

	# hand ranks
	img = cv2.rectangle(img, (525, 1293), (560, 1330), (255, 255, 0), 2)
	img = cv2.rectangle(img, (600, 1285), (632, 1325), (255, 255, 0), 2)
	cv2.imwrite('images/all_bboxes.png', img)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Preprocess large image for object detection')
	parser.add_argument("--image", "-i", required=True, help="Path to input image")

	args = parser.parse_args()

	img = cv2.imread(args.image)
	table_s, table_r = get_table_cards(img)
	hand_s, hand_r = get_hand_cards(img)
	balances = get_balances(img.copy())

	bets = get_bets(img.copy())

	pot = get_pot(img.copy())

	result = {'balances': balances,
			  'bets': bets,
			  'pot': pot,
			  'hand': {
				  'suit': hand_s,
				  'rank': hand_r
			  },
			  'table':{
				  'suits': table_s,
				  'ranks': table_r
			  }}

	with open('result.json', 'w') as fp:
		json.dump(result, fp)

	# print_all_bboxes(img)

