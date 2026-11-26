# Локализация Unitree Go2 на 3D-карте помещения

Первый проезд робота даёт карту. Во втором проезде робот самостоятельно находит своё начальное положение и в реальном времени публикует позу корпуса в координатах карты. Решение работает с записями ROS 2 Humble. Начальную позу вводить не требуется. Узел получает сканы /utlidar/cloud_deskewed, одометрию /utlidar/robot_odom и измерения гироскопа /utlidar/imu (около 250 Гц).

## Результат

На 1 480 общих кадрах 90,57% точек следующего скана выше пола попали в пределах 20 см от карты. При полном ICP результат составил 90,44% с 79 резкими поправками, у решения с фильтром таких поправок было ноль. Метрика описывает совпадение наблюдений с картой. Независимого эталона позы для второго проезда нет.

На штатном старте получена поза на 45-м скане и далее на каждом из 1 481 кадров с момента захвата. В трёх полных прогонах от первого облака до первой позы прошло 3,176–3,196 с. За 72 мс пришли 4 441 из 4 443 поз. Объединённая задержка P99 равна 12,77 мс, P99,9 — 30,78 мс; максимум — 170,15 мс. [Сравнение методов](results/metrics.csv) · [Замеры каждого прогона](results/timing-runs.csv) · [Места резких поправок](results/icp-jumps.png)

## Воспроизвести решение

Нужны Docker, Python 3.11 для чтения MCAP на хосте и выданные организаторами записи. Скопируйте распакованные папки с metadata.yaml и MCAP в data/rosbags/bag_for_mapping и data/rosbags/bag_for_localization. Источником базового образа служит [стенд организаторов](https://github.com/ivr0ma/hackathon/tree/ecbeeb681f7fe1e274402623da27444f63a4c61a/hackathon). В каталоге стенда рядом с Dockerfile, play_viz.sh и go2.rviz выполните:

```sh
docker build -t hakatonius:latest .
```

Следующие команды выполняются из корня репозитория команды. Соберите карту на хосте и слой с SciPy для контейнера. Для Linux/AMD64 замените manylinux2014_aarch64 на manylinux2014_x86_64.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-host.txt
.venv/bin/python analysis/final_eda/build_map.py

wheel_dir=$(mktemp -d)
.venv/bin/python -m pip download --only-binary=:all: --no-deps \
  --platform manylinux2014_aarch64 --implementation cp --abi cp310 \
  --python-version 3.10 --dest "$wheel_dir" scipy==1.10.1
docker build --network none --pull=false -f solution_ros/Dockerfile \
  -t mgts-localization:local "$wheel_dir"

bash solution_ros/run_replay.sh runs/final \
  --method multiview-ekf --ekf-profile synchronized --prefetch-start --gyro
.venv/bin/python solution_ros/check_result.py \
  data/rosbags/bag_for_localization runs/final
```

Короткую проверку можно сделать с отдельной выходной папкой и параметром `--duration 12`. Для полного повторного прогона укажите новое имя папки. В runs/final/result_bag будут metadata.yaml и MCAP. В среде ROS 2 воспроизведите их командой `ros2 bag play runs/final/result_bag --clock`. В RViz задайте Fixed Frame `map` и добавьте /solution/map, /solution/scan, /solution/path и /tf. Внешний наблюдатель записывает задержки в observer.csv. Для проверки геометрии позы и TF используется verification.json.

## Как читать код

1. [build_map.py](analysis/final_eda/build_map.py) проходит по /utlidar/cloud_deskewed первой записи. Облака уже приведены в локальные оси этого проезда. Убираем нулевые и нечисловые точки, усредняем координаты внутри кубиков со стороной 5 см. Получаем 163 163 точки в map.npz.
2. [localize.py](solution_ros/localize.py), методы on_cloud и process_ready, принимают облако и /utlidar/robot_odom второго проезда. Узел дожидается двух соседних сообщений одометрии и интерполирует положение и ориентацию корпуса на время скана.
   [gyro_motion.py](analysis/localization/gyro_motion.py) интегрирует угловые скорости за интервал между сканами и переводит их в поворот курса корпуса. При разрыве потока используется одометрия. [Калибровка осей и смещения](analysis/localization/calibrate_gyro.py) выполнена на первом проезде.
3. [runtime_method.py](analysis/localization/runtime_method.py), метод update, переключает узел со стартового поиска на сопровождение. В [cold_start.py](analysis/localization/cold_start.py) сравниваются положения и направления на карте. Первые десять сканов блока из пятнадцати уточняют кандидатов, следующие пять проверяют. Фоновый поток заранее выполняет тяжёлую часть поиска. При согласии двух проверок принимается стартовая привязка.
4. [geometry.py](analysis/localization/geometry.py) ищет соответствия по карте и оценивает поправку ICP. В [kalman_tracking.py](analysis/localization/kalman_tracking.py) фильтр прогнозирует перемещение по одометрии. Для прогноза курса он поровну учитывает поворот по одометрии и по гироскопу, затем уточняет позу по совмещению до четырёх облаков. Измерение привязано к центру наблюдаемых точек. Если скан слабый, поза продолжает обновляться по прогнозу. От карты оцениваются горизонтальное положение и курс; высоту, крен и тангаж даёт бортовая одометрия.
5. [localize.py](solution_ros/localize.py), метод process, публикует /solution/pose, /tf (map → odom → base_link), карту, текущий скан, путь и статус. [observe_latency.py](solution_ros/observe_latency.py) измеряет интервал от получения облака до получения позы с той же меткой на одних монотонных часах. [check_result.py](solution_ros/check_result.py) повторно читает bag и сверяет выход с исходными кадрами и TF.

## Материалы сдачи

Откройте [презентацию](mgts-go2.pdf). Для показа без сети заранее скачайте и откройте PDF и [видео финального решения](results/third.mp4) в локальном плеере. В папке results лежат результирующий rosbag2, оба ракурса RViz и покадровые замеры. [Инструкция по материалам](results/materials.md)

На 320 кадрах с поворотом доля точек в пределах 20 см от карты с гироскопом составила 89,071%, с одной одометрией в прогнозе фильтра — 88,959%. [Парное сравнение методов](results/gyro-geometry.json) · [Изменение траектории](results/gyro-trajectories.png)

Метрики геометрии реализованы в [metrics.py](analysis/validation/metrics.py). Их модульные проверки используют NumPy и SciPy. В окружении хоста с установленным SciPy запустите `python -m unittest discover -s analysis/validation -p test_metrics.py`. Для основного запуска SciPy устанавливается внутрь Docker.

## Компоненты

Картирование, поиск старта, ICP, EKF, узел ROS и проверки реализованы кодом команды. Для массивов используется NumPy (BSD-3-Clause), для деревьев ближайших соседей, поворотов и полей расстояний — SciPy (BSD-3-Clause), для чтения и проверки записей — rosbags (Apache-2.0). ROS 2 Humble и форматы сообщений предоставляются стендом организаторов. Экранные записи сделаны в RViz.
