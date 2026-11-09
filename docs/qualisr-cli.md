# Проверка CLI QualiSR-Lab 0.2.0

## Воспроизведение

Из корня репозитория выполните команды ниже. Проверка прошла на Python 3.12.16 с установленным [QualiSR-Lab 0.2.0](https://pypi.org/project/qualisr-lab/0.2.0/).

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install 'qualisr-lab[regressors]==0.2.0'
.venv/bin/python -m pip check
.venv/bin/qualisr-run-regressors --help
.venv/bin/python tests/smoke_qualisr_cli.py
```

[Smoke-тест](../tests/smoke_qualisr_cli.py) печатает путь к новой временной директории. В ней остаются тестовые CSV, `split/config.json`, `cv/config.json`, логи и результаты. Каждый запуск выполняется отдельной командой `qualisr-run-regressors --config <путь>` с одним Random Forest на 16 деревьях и с `seed=42`, `split_seed=42`. Для запуска достаточно локальных CSV.

## Входные данные и конфиг

В `data/labels.csv` содержатся 120 оценок MOS для 40 исходных изображений и трёх SR-методов. Standalone-загрузчик читает `method`, `test_case` и `score`. Поле `image` в этом режиме не используется. Из `method` и `test_case` загрузчик собирает ключ `QualiSR-Set120/<Method>/<test_case>.npy.gz`. Имена `pasd`, `realesrgan` и `supir` приводятся к `PASD`, `RealESRGAN` и `SUPIR` соответственно.

В исходных локальных CSV ключ имел вид `PASD:<test_case>`. При прямой загрузке `nr.csv` версия 0.2.0 сообщала об отсутствии всех 120 ожидаемых ключей. На этапе 9.B локальные таблицы признаков переведены на ключ `QualiSR-Set120/<Method>/<test_case>.npy.gz`. В `stats.csv` колонка `name` переименована в `sample_id`. Остальные значения в CSV сверены с исходными файлами. Тестовый набор теперь копирует готовые `labels.csv`, `nr.csv` и `fr.csv` и проверяет полное совпадение ключей без дубликатов.

[Шаблон standalone JSON](../configs/templates/regressors.json) содержит `models` верхнего уровня. Отсутствие `datasets` включает загрузку по `paths.labels` и `paths.features_root`. При генерации конфига сервис подставляет абсолютные пути к подготовленным CSV и отдельный `paths.plots_root` для запуска. `features.feature_files` сопоставляет группы с путями, а `features.include` и `fr_refs` выбирают обучающие признаки. В smoke-тесте модель использует NR и FR с reference `rlfn`, а `correlation_metrics.items` берёт MUSIQ из NR и `lpips-vgg_gt` из исходного FR CSV. Для LPIPS указано `higher_is_better: false`.

### Ограничение `test_size`

В режиме `cross_validation.enabled=false` функция `load_feature_bundle_samples` присваивает всем строкам `regressors.test_size=0.2`. `split_dataset` читает это значение из записей датасета и запускает `GroupShuffleSplit` с `split_seed` и группировкой по `test_case`. В проверке `load_config_with_samples` поля `test_size=0.5` верхнего уровня и внутри `cross_validation` оставили фактическое значение `0.2` у всех 120 строк. `--help` также не содержит флага для этой настройки. Ветка `datasets` требует файлы SR и LR.

Сервисный API принимает `test_size=0.2` и отклоняет другие значения с HTTP 422. `split_seed` остаётся настраиваемым. В CV проверена настройка `cross_validation.n_splits`, группировка по `dataset/test_case` и перемешивание групп с `split_seed`.

## Артефакты и проверка выборок

Корень артефактов CLI имеет вид `<plots_root>/<experiment_name>@pca<pca_n>`. Суффикс появляется даже при выборе только NR и FR. Пути ниже указаны относительно этого корня.

| Артефакт | Split | CV |
|---|---|---|
| `correlations/correlations.csv` | PLCC и SRCC на тестовой выборке | Среднее и выборочное стандартное отклонение корреляций по фолдам, а также `n_folds` |
| `predictions/predictions_randomforest.csv` | `sample_id`, `test_case`, `mos`, `prediction` и служебные поля | OOF-предсказания с дополнительным полем `fold` |
| `correlations/correlations_by_group.csv` | Детали расчёта, включая `n_samples` | Отдельно в каждом `fold_XX/` |
| `correlations/cross_validation_folds.csv` | | Строки PLCC/SRCC по фолдам |
| `metadata/cross_validation_folds.csv` | | Назначение `sample_id` на `fold` |
| `correlations/correlations.png` | График корреляций | График средних и отдельный график в каждом `fold_XX/` |
| `predictions/mos_prediction_scatter_all_regressors.png` | MOS и предсказания | Отдельно в каждом `fold_XX/` |
| `importances/importance_randomforest.csv` и `.png` | Важности признаков | Отдельно в каждом `fold_XX/` |
| `metadata/config.json`, `metadata/dataset_snapshot.csv`, `log.txt` | Разрешённый конфиг, таблица признаков и лог CLI | Те же артефакты в корне. Конфиг и snapshot также есть в каждом `fold_XX/` |

В обоих режимах под `per_dataset/QualiSR-Set120/` есть дополнительная разбивка по датасету и `metadata/validation_samples.csv`. В режиме CV эта разбивка находится внутри каждого фолда. Входной JSON, переданный в CLI, хранится отдельно от `metadata/config.json`.

В исходном JSON сервиса раздел `provenance` фиксирует версию QualiSR-Lab, версию Python, версии NumPy, pandas, SciPy, scikit-learn и выбранных XGBoost или CatBoost, а также размеры и SHA-256 файлов датасета.

Split дал 24 тестовых строки из 8 групп. CV дал четыре фолда по 30 строк из 10 групп и ровно одно OOF-предсказание для каждой из 120 строк. Тест сопоставляет предсказания с ожидаемыми `sample_id` и проверяет, что варианты одного `test_case` попадают в одну часть split или в один validation fold.

Для MUSIQ и LPIPS-VGG+GT значения конечны на всех проверенных строках. Тест сверяет `n_samples` с количеством предсказаний и независимо пересчитывает PLCC и SRCC по тем же `sample_id` для каждой из двух метрик и регрессора. CV-сводка совпала со средним по фолдам. При импорте других данных пропуски в baseline могут уменьшить `n_samples`. Для проверки сопоставимости используйте `correlations_by_group.csv` соответствующего split или фолда и сверяйте покрытие по `sample_id`.
