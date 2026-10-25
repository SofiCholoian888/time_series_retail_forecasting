# Retail Demand Forecasting

Учебный проект прогнозирования ежедневных продаж по парам «магазин — товар» на 28 дней. Основной запускаемый pipeline: **XGBoost + MLForecast**, рекурсивный прогноз, временная валидация и пакетный инференс.

**Начать: [пошаговая инструкция](START_HERE_RU.md) → [ноутбук Google Colab](notebooks/00_COLAB_RUN_ALL.ipynb).**

## Три домашних задания

| ДЗ | Что проверять |
|---|---|
| 1. Рефакторинг и окружение | `src/retail_forecasting/`, `tests/`, `pyproject.toml`, `poetry.lock`, `poetry.toml`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml` |
| 2. Model Card и место модели | [model_card.md](model_card.md), [Model_BPMN.md](Model_BPMN.md) |
| 3. Процесс автоматического переобучения | [Model_retrain_BPMN.md](Model_retrain_BPMN.md) |

Результат проверки исходного репозитория: [AUDIT_RU.md](AUDIT_RU.md). Фактически выполненные проверки исправленной версии: [docs/VALIDATION.md](docs/VALIDATION.md).

## Данные и задача

Источник: [Kaggle Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data), `train.csv`: `date,store,item,sales`. Историческое описание проекта: 913 000 наблюдений, 500 рядов, 01.01.2013–31.12.2017. Исходные данные в Git не включаются; для проверки качества загрузите свой `train.csv`.

Чистый технический пример создаётся скриптом на синтетических данных. Он проверяет работу кода, а его качество нельзя переносить на Kaggle или бизнес.

## Быстрый локальный запуск

Нужны Python 3.11 или 3.12, Git и Poetry 2.2.1. Для Colab всё устанавливается ноутбуком.

```bash
poetry check --lock
poetry install
poetry run pre-commit install
poetry run pre-commit run --all-files
poetry run pytest -q
poetry run python scripts/make_demo_data.py
poetry run retail-forecast evaluate --data data/raw/demo.csv --config configs/demo.json --output results/demo
poetry run retail-forecast train --data data/raw/demo.csv --config configs/demo.json --model-dir artifacts/demo-v1
poetry run retail-forecast predict --model-dir artifacts/demo-v1 --output results/demo/forecast.csv
```

При повторном `train` задайте **новую** папку версии (`demo-v2`): существующая модель не перезаписывается. После загрузки реального CSV используйте `configs/default.json`, `--expected-series 500` и отдельные папки `results/full` / `artifacts/full-v1`.

## Архитектура

| Компонент | Ответственность |
|---|---|
| `data/validation.py` | Схема, типы, конечность значений, неотрицательность, дубли, пропуски дат, выровненная панель |
| `features/specification.py` | Лаги 1/7/14/28, три скользящих средних, календарные и статические признаки |
| `models/forecast.py` | XGBoost, LocalStandardScaler, рекурсивная стратегия, единая обработка выхода |
| `evaluation/` | sMAPE/MAE/RMSE, rolling-origin, SeasonalNaive, сегментные метрики |
| `pipeline.py`, `cli.py` | Команды `evaluate`, `train`, `predict` |
| `utils/metadata.py` | SHA-256 входа/исходников/артефактов, конфигурация, версии библиотек, Git commit |

`.venv/` создаётся локально и игнорируется Git. В Git фиксируются **описание окружения и lock-файл**, а не установленные пакеты. Пустые `__init__.py` допустимы: полезная логика находится в соседних модулях.

## Протокол оценки

`configs/default.json`: горизонт 28, 3 окна, шаг 28, `refit=False`. На первом срезе обучается глобальная модель, затем используются обновлённые доступные наблюдения между срезами без полного переобучения. Внутри 28-дневного окна прогноз рекурсивный: фактические продажи после cutoff недоступны.

На каждом cutoff SeasonalNaive повторяет последнюю **известную** неделю; будущие факты не используются даже на шагах 8–28. Проверяются точные ключи `unique_id,cutoff,ds`, целевые значения и полнота результата. Оценка и инференс одинаково ограничивают окончательные прогнозы снизу нулём без округления; внутренние рекурсивные шаги MLForecast сохраняют исходную стратегию.

Команда `evaluate` записывает `metrics.csv`, `segment_metrics.csv`, `cv_predictions.csv`, `evaluation.json`. Команда `train` отдельно обучает модель на всей доступной истории и сохраняет её со scaler и состоянием рядов. Это сохранение кандидата, **без автоматического решения о публикации**.

## Исторические результаты

В исходном репозитории опубликованы XGBoost: **sMAPE 12.975%, MAE 6.168, RMSE 8.101**; AutoETS: **16.978%, 8.329, 10.888**. Это результаты прежнего исследования. Они **не были заново подтверждены на полном датасете при рефакторинге** и не являются метриками исправленной версии. Полная историческая таблица и условия: [Model Card](model_card.md).

Для новой версии используйте только `metrics.csv` собственного полного запуска и сохраните `evaluation.json` рядом. При изменении версий библиотек, признаков или обработки выхода прежние числа нельзя автоматически приписывать новому артефакту.

## Исследовательские ноутбуки

Исходные три ноутбука сохранены в [notebooks/research/](notebooks/research/), с исправленными путями и проверками результатов. Их Task 1/2/3 — **этапы исследования**, а не номера этих домашних заданий. Для трёх ДЗ достаточно основного pipeline и документации.

Для полного повторения EDA, пяти статистических моделей, трёх ML и трёх DL:

```bash
poetry install --with research
poetry run python -m ipykernel install --user --name retail-forecast --display-name "Retail forecasting (Poetry)"
poetry run jupyter lab
```

Выберите kernel этого окружения и выполните 01 → 02 → 03. Данные ожидаются в `data/raw/train.csv`. Результаты исследования лежат отдельно в `results/research/`. Полный DL-прогон при подготовке этой версии не выполнялся. Его фиксированная группа зависимостей включена в lock-файл; совместимость всей исследовательской ветки требует отдельного запуска. CI проверяет поддерживаемый пакет и новый Colab-ноутбук; исследовательские notebooks исключены из Ruff, но проверяются на корректность структуры и очищаются от outputs.

## Граница реализации

Работают локальный/Colab запуск, проверки данных, backtest, сериализация, повторный инференс и CI-конфигурация. ДЗ №3 описывает будущую автоматизацию: планировщик, текущий источник продаж, champion/challenger gate, production registry, переключение активной версии, мониторинг и откат ещё требуют интеграции.
