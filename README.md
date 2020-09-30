## Парсинг покерного стола

Для расознавания чисел/букв была определена область в которой находятся символы, 
затем с помощью библиотеки OpenCV в этой области находятся контуры, которые помещаются в ограничивающий прямоугольник
Изображение в ограничивающем прямоугольнике подаётся на вход свёрточной нейросети, обученной на наборе EMNIST
(в наборе перед обучением были удалены все буквы кроме A, K, Q, J)
все распознанные цыфры из одной области записываются в одно 

## Запуск скрипта распознавания:
```
python3 main.py -i 'images/image.png'
```
## Результат работы скрипта
Весь результат в result.json в корневой папке проекта. Поля: balances - баланс игроков, bets - ставки игроков, pot - общий банк, hand - карты на руке (suit/rank масть/ранг соответственно), table - аналогично с hand