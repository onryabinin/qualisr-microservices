# qualisr-microservices

Проект основан на моей научной работе [QualiSR-Lab](https://github.com/msu-video-group/QualiSR-Lab), статья отправлена в журнал IEEE Access.

Постановка задачи, архитектура пяти сервисов, контракт CSV, публичный API и сценарий защиты собраны в [итоговом отчёте](docs/report.md).

## Запуск

```bash
docker compose up --build -d --wait
docker compose ps
```

Откройте интерфейс и выберите QualiSR-Set120 или импортируйте совместимый набор CSV. Выберите признаки, регрессоры, seed модели, seed разбиения и групповой split или CV. Web Master публикует OpenAPI. Внешние порты можно изменить в `.env` по примеру `.env.example`. [Интерфейс](http://127.0.0.1:8501) · [OpenAPI](http://127.0.0.1:8080/docs)

Collector регистрирует встроенный набор из `data/` при первом старте. Storage хранит датасеты, SQLite и результаты в именованных томах. ML Service запускает один эксперимент за раз через CLI QualiSR-Lab 0.2.0. Предсказания относятся к тестовой выборке или CV-фолдам. [Формат CLI и артефактов](docs/qualisr-cli.md)

## Результаты в интерфейсе

По завершении запуска доступны PLCC/SRCC регрессоров и IQA-метрик, интерактивный график MOS и предсказаний, журнал, важность признаков и скачивание исходного JSON и CSV. В режиме CV таблица показывает среднее и выборочное стандартное отклонение корреляций по фолдам. Отдельная таблица содержит оценки каждого фолда, а график показывает OOF-предсказания.

В истории отметьте завершённые запуски и нажмите «Сравнить выбранные». Интерфейс проверяет датасет, режим, набор `sample_id` и принадлежность к фолдам. Строки IQA-метрик с неполным покрытием выделяются отдельно. Если эксперимент остался в статусе «Создан», откройте его в истории и нажмите «Поставить в очередь».

## Импорт и запуск через API

Загрузите `labels.csv` и один или несколько файлов признаков. Имя каждого файла задаёт группу признаков, например `nr.csv` → `nr`. Collector проверяет, что оценки MOS конечны, а `sample_id` уникальны и совпадают с набором ключей, который QualiSR-Lab получает из `method` и `test_case`. В отчёте импорта сохраняется `import_id`. Для просмотра журнала и чтения батчей доступны `/imports/{import_id}` и `/datasets/{dataset_id}/batches?offset=0&limit=10&feature=nr`. Без `feature` возвращаются строки labels. Смещение за пределами датасета даёт пустой батч.

```bash
curl -fsS http://127.0.0.1:8080/datasets/import \
  -F name=example -F labels=@data/labels.csv \
  -F features=@data/features/nr.csv -F features=@data/features/fr.csv

curl -fsS http://127.0.0.1:8080/experiments \
  -H 'Content-Type: application/json' \
  -d '{"dataset_id":"qualisr-set120","mode":"cv","n_splits":4}'

# Подставьте id из ответа POST /experiments.
run_id=YOUR_RUN_ID
curl -fsS -X POST "http://127.0.0.1:8080/experiments/$run_id/run"
curl -fsS "http://127.0.0.1:8080/experiments/$run_id"
```

После статуса `succeeded` доступны `/experiments/{run_id}/results` и `/experiments/{run_id}/artifacts`. Отдельный артефакт скачивается через `/artifacts/{artifact_id}`. Журнал процесса доступен в `/experiments/{run_id}/log`.

По умолчанию используются доступные группы `nr` и `fr`. Если в датасете есть другие группы, их можно выбрать полем `features`, например `["stats", "resnet_pca5"]`. Для FR по умолчанию используется RLFN. Поле `fr_refs` принимает список разных референсов, для каждого из которых есть подходящая колонка. В grouped split версия 0.2.0 фиксирует долю тестовых групп на `0.2`. CV принимает `n_splits`. Неизвестные поля настроек получают HTTP 422.

Без поля `models` запускается Random Forest на 16 деревьях. Чтобы выбрать несколько регрессоров, передайте объект `models`, например `{"randomforest":{"n_estimators":32},"xgb":{"max_depth":3},"catboost":{"iterations":50}}`. Форма показывает поддерживаемые параметры и их значения по умолчанию. API проверяет имена, типы и диапазоны параметров перед созданием запуска. API предсказаний для произвольных новых данных и сохранение обученных моделей выходят за рамки этого сценария.

## Демонстрация и сквозная проверка

После запуска Compose выполните из корня репозитория следующие команды.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/demo.py
.venv/bin/python tests/smoke_microservices.py
```

Демонстрация запускает Random Forest на QualiSR-Set120 в режимах split и CV, печатает PLCC/SRCC и адреса артефактов. Для показа одного режима передайте `--mode split` или `--mode cv`.

Тест запускает split на встроенном наборе и CV с тремя регрессорами на загруженной копии. Он проверяет PLCC/SRCC, совпадение оценочных ID и фолдов между регрессорами, валидацию гиперпараметров, батчи, ошибки импорта и скачивание артефактов. Для изменённых портов укажите `QUALISR_API_URL` и `QUALISR_UI_URL`. Команды `docker compose stop` и `podman compose stop` останавливают сервисы с сохранением данных в томах.

### Ошибки, очередь и сохранность данных

Для проверки создайте отдельный Compose-проект. Порты в примере должны быть свободны. Тест добавляет временные датасеты и запуски в тома этого проекта.

```bash
export QUALISR_API_PORT=28080 QUALISR_UI_PORT=28501
export QUALISR_API_URL=http://127.0.0.1:28080 QUALISR_UI_URL=http://127.0.0.1:28501
snapshot_dir=$(mktemp -d)
docker compose -p qualisr-check up --build -d --wait
.venv/bin/python tests/stability_microservices.py "$snapshot_dir/state.json"
docker compose -p qualisr-check up --force-recreate --no-build -d --wait
.venv/bin/python tests/stability_microservices.py "$snapshot_dir/state.json" --verify
docker compose -p qualisr-check stop
```

Первая фаза проверяет ответы на невалидные запросы, два варианта отказа CLI, одновременную постановку запусков в очередь и изоляцию результатов. Вторая сверяет метаданные датасета, журнала импорта, истории и артефактов, а также SHA-256 скачанных файлов после пересоздания контейнеров.
