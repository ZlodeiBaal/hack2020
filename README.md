# hack2020 repo, 427 team

Документация к коду репозитория команды 427:

## Модули (папка src):
- quality_estimator.py - Главный модуль предиктивной модели.
- data.py - Модуль обработки данных. (Первичная генерация данных, чтение и разделение на трейновые и тестовые датасеты)
- dpipe_metrics.py - Модуль несущих метрик, таких как (dice_score, iou, surface_distances, assd, centres_distance и других). А так же реализация алгоритма выбора пересекающихся (являющихся близкими в терминах некоторой метрики) инстансов (get_matching).
- matching_metrics.py - Модуль метрик, подсчитывающих различные аггрегированные статистики над матчингами (соответствиями между инстансами объектов). Пример подобных метрик: **match2tpr** - количество верно найденных (например покрытых по iou > 0.5) объектов из класса Expert деленное на общее количетсво объектов этого класса на картинке.
- model_selection: служебный модуль, содержит реализованную кастомную функцию *cross_val_score* для валидации моделей со структурой данных (X, Xy, y)

## Пайплайн для работы с алгоритмом:

Пайплайн полного воспроизведения работы находится в файле **notebooks/427_solution.ipynb**. В данном ноутбуке показан и прокомментирован полный цикл работы с данными от распаковки сырого архива и подготовки трейновой и тестовой выборки до подготовки финального файла с предсказаниями, включая полный пайплайн обучения алгоритма, выбор признаков и предсказания на тестовых данных.

## Основные классы и методы:

### BaseQualityEstimator

Основной фреймворк для разработки алгоритма оценки качества разметки - это класс **BaseQualityEstimator**. Данный класс реализован по принципу scikit-learn-like. В нем есть методы fit, predict / predict_proba. Как и в любой sklearn-модели инициализация параметров алгоритма происходит во время инициализации объекта класса, параметрами модели служат наборы метрик 3 типов: **metrics** - простейшие метрики для сравнения 2d/3d сегментационных масок (такие как dice_coefficient и пр.), **unary_metrics** - метрики, которые позволяют оценивать свойства единственной 2d/3d сегментационной маски без референсной картинки (такие как площадь, круглость и прочие), **matching_metrics** - метрики, которые позволяюх сравнивать сегментационные карты с разделением по инстансам (например, **match2tpr** - количество верно найденных (например покрытых по iou > 0.5) объектов из класса Expert деленное на общее количетсво объектов этого класса на картинке.)

#### Процесс обучения (основной метод данного класса) можно разделить на несколько этапов:
* Подсчет простых метрик из классов **metrics** и **unary_metrics** на всех трейновых семплах
* Получение пар матчингов (**get_matching**) между всеми инстансами GT и предсказаний.
* Подсчет сложных метрик (**matching_metrics**) по сеткам параметров и многостадийная аггрегация (между инстансами, которые соответствуюх одному инстансу GT-картинки и наоборот, потом между всеми инстансами семпла).
* Обучение мета-алгоритмов (набор нескольких слабо скоррелированных моделей, который определяется при инициализации **BaseQualityEstimator**) на конкатенации всех метрик, описанных выше (**metrics**, **unary_metrics**, **matching_metrics**) с параллельным отбором признаков для каждого из алгоритмов с помощью [RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html)
* Сохранение обученных алгоритмов и отобранных признаков внутри объекта класса для дальнейших предсказаний.

#### Процесс предсказания:
* Подсчет простых метрик из классов **metrics** и **unary_metrics** на всех трейновых семплах
* Получение пар матчингов (**get_matching**) между всеми инстансами GT и предсказаний.
* Подсчет сложных метрик (**matching_metrics**) по сеткам параметров и многостадийная аггрегация (между инстансами, которые соответствуюх одному инстансу GT-картинки и наоборот, потом между всеми инстансами семпла).
* Предсказание на выбранном поднаборе наборе признаков каждым из мета-алгоритмов и их аггрегация (взвешивание).


### get_matching:

Оптимизированная функция поиска мэтчингов между инстансами GT и PRED изображений. Возвращает 4 листа из мэтчингов [GT, [PRED]], [PRED, [GT]], unmatched GT, unmatched PRED. Позволяет дальше создавать произвольные сложные метрики на инстансах (примеры - в **matching_metrics**).

