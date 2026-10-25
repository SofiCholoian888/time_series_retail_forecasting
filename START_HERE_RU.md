# Как запустить проект и загрузить исправления на GitHub

## 1. Что находится в архиве

Распакуйте архив. Внутри — папка `time_series_retail_forecasting` с кодом, документацией и ноутбуками. Все содержимое этой папки соответствует корню репозитория, а не вложенной папке внутри него.

| Файл / папка | Для чего |
|---|---|
| `AUDIT_RU.md` | Что было правильно и что требовало исправления |
| `notebooks/00_COLAB_RUN_ALL.ipynb` | Основной пошаговый запуск в Google Colab |
| `src/retail_forecasting/` | Рабочий Python-пакет |
| `tests/` | Проверки корректности |
| `model_card.md`, `Model_BPMN.md`, `Model_retrain_BPMN.md` | Файлы для сдачи ДЗ №2–3 |
| `pyproject.toml`, `poetry.lock`, `poetry.toml` | Описание и воспроизведение окружения |
| `.pre-commit-config.yaml`, `.github/workflows/ci.yml` | Автоматические проверки |
| `notebooks/research/` | Три исходных исследовательских ноутбука с исправлениями |
| `docs/VALIDATION.md` | Какие проверки действительно выполнены при подготовке |

Папки `.venv` и `.git`, реальные данные и обученные модели в исходный ZIP не включены. Пустые `__init__.py` можно оставить: это нормально для Python-пакетов.

## 2. Google Colab: самый простой запуск

1. Сохраните исправленный ZIP и распакуйте его на компьютере. Сам ZIP тоже оставьте: он потребуется для загрузки исходников в Colab.
2. Откройте [Google Colab](https://colab.research.google.com/).
3. Выберите **Файл → Открыть блокнот → Загрузить**. Выберите файл `notebooks/00_COLAB_RUN_ALL.ipynb`. Если ноутбук выдан отдельно, можно загрузить его напрямую.
4. В меню **Среда выполнения → Сменить среду выполнения** выберите CPU / без аппаратного ускорителя. GPU для основного XGBoost pipeline не нужен.
5. Выполните первую ячейку кнопкой ▶. Когда появится кнопка выбора файла, загрузите **исправленный ZIP проекта**.
6. Выполните ячейку установки. Она установит Poetry 2.2.1 и зависимости строго из `poetry.lock` в отдельную `.venv`. Если версия Python в Colab не подходит, ноутбук установит отдельный Python 3.12. Kernel Colab остаётся прежним; код проекта запускается отдельным процессом.
7. Выполните ячейку проверок. Ожидается успешное завершение pre-commit, pytest и сборки пакета. Если hook автоматически поправил форматирование, notebook повторит проверку. При другой ошибке он остановится, а не напечатает ложное «успех».
8. Выполните технический пример. Он создаёт 4 синтетических ряда, вычисляет временную оценку, обучает модель, сохраняет её и загружает для прогноза. Эти метрики нужны для проверки работы кода и не заменяют настоящий эксперимент.
9. Для оценки на полном наборе откройте ячейку `RUN_FULL = False`, измените значение на `True`, запустите и загрузите **train.csv** из [Kaggle Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data). Нужны поля `date,store,item,sales`. Не выбирайте `test.csv`: в нём нет целевой переменной.
10. Выполните последнюю кодовую ячейку. Скачиваются два архива: `retail_source_for_github.zip` и `retail_run_results.zip`. Если браузер блокирует второй download, разрешите несколько скачиваний или откройте слева панель «Файлы», папку `/content/retail_hw_run` и скачайте архив вручную через меню файла.

Colab удаляет временные файлы после завершения сессии. Скачайте результаты до закрытия. Исходный датасет в ZIP результатов не копируется; храните `train.csv` отдельно. При повторении обучения выполняйте ячейку технического примера заново: она создаёт новый RUN_ID, чтобы прежняя модель не перезаписывалась.

## 3. Как понять результаты

| Результат | Что смотреть |
|---|---|
| `results/full_<run_id>/metrics.csv` | Новые sMAPE, MAE и RMSE XGBoost и SeasonalNaive |
| `results/full_<run_id>/cv_predictions.csv` | Реальные out-of-sample прогнозы и факты; 42 000 строк для исходных 500 рядов |
| `results/full_<run_id>/segment_metrics.csv` | Ошибки по рядам, магазинам, товарам, cutoff и горизонту |
| `results/full_<run_id>/evaluation.json` | Даты, параметры, версии и hash датасета; сохраняйте вместе с метриками |
| `results/full_<run_id>/forecast.csv` | Прогноз после обучения на всей истории; 14 000 строк для 500 рядов и 28 дней |
| `artifacts/full_<run_id>/` | Сохранённая модель, scaler, состояние рядов и metadata |

На настоящем датасете дата последнего факта — 31.12.2017. Поэтому будущий прогноз начинается **01.01.2018**, а не сегодня. Модель не умеет «перепрыгнуть» девять лет без истории.

Если новые значения отличаются от 12.975 / 6.168 / 8.101, это само по себе не означает поломку. Эти числа были опубликованы для старого окружения и протокола. У новой версии закреплены другие версии библиотек и явно задана обработка отрицательного выхода. Указывайте фактически полученные результаты и сохраняйте конфигурацию. Не заменяйте их числами из README.

Основной pipeline сравнивает выбранный XGBoost с SeasonalNaive. Полное повторение прежнего сравнения 11 алгоритмов — дополнительное исследование, которое запускается отдельно и может быть существенно дольше.

## 4. Загрузка на GitHub через GitHub Desktop

Этот способ сохраняет историю репозитория и помогает увидеть каждое изменение. Веб-интерфейс GitHub **не распаковывает ZIP**: загрузить только ZIP недостаточно.

1. Установите [GitHub Desktop](https://desktop.github.com/) и войдите в свой GitHub-аккаунт.
2. Выберите **File → Clone repository → URL**. Вставьте `https://github.com/SofiCholoian888/time_series_retail_forecasting`. Выберите папку на компьютере и нажмите **Clone**. Если уже есть рабочая локальная копия, сначала сохраните её незакоммиченные изменения и сделайте **Fetch origin / Pull origin**.
3. Выберите **Branch → New branch**. Назовите ветку `homework-fixes` и создайте её от актуальной `main`.
4. Выберите **Repository → Show in Explorer** (на macOS — Finder).
5. Скопируйте **содержимое** папки `time_series_retail_forecasting` из исправленного архива в корень клонированного репозитория, согласившись на замену одноимённых файлов. Не создавайте вложенную одноимённую папку. Сохраните существующую `.git`: она содержит историю и связь с GitHub.
6. Исследовательские notebooks перенесены. Удалите из старого расположения только эти три файла, если они остались после копирования: `notebooks/01_eda.ipynb`, `notebooks/02_statistical_models.ipynb`, `notebooks/03_ml_dl_anomaly_detection.ipynb`. Их исправленные копии теперь находятся в `notebooks/research/`. Не удаляйте новые notebooks или чужие файлы.
7. Включите отображение скрытых файлов и убедитесь, что скопировались `.github/`, `.gitignore` и `.pre-commit-config.yaml`. В списке изменений не должно быть `.venv/`, `train.csv`, `.git/`, токенов, `kaggle.json`, бинарных моделей или ZIP результатов.
8. Вернитесь в Desktop. В **Summary** введите `Refactor forecasting pipeline and align homework documentation`. Проверьте diff и нажмите **Commit to homework-fixes**.
9. Нажмите **Publish branch**, затем **Create Pull Request / Preview Pull Request**. Заголовок: `Refactor forecasting pipeline and correct model documentation`. В описании кратко укажите модульный pipeline, тесты, Poetry/pre-commit и исправление рекурсивной стратегии.
10. Откройте **Actions** или вкладку checks Pull Request. Дождитесь успешного workflow **CI**. Именно этот новый запуск подтверждает, что проверки выполнились на GitHub. Если workflow не стартует, убедитесь, что `.github/workflows/ci.yml` находится в корне репозитория и Actions разрешены в настройках.
11. После успешных проверок выполните **Merge pull request → Confirm merge**. Ссылки на `blob/main/...` теперь показывают исправленные файлы.

GitHub Desktop использует собственную авторизацию. Не вставляйте Personal Access Token в URL репозитория, код Colab, `.git/config`, архивы или сообщения.

При последующих локальных коммитах с установленным pre-commit держите `.venv` доступной. Если вы переносите папку проекта, переустановите hook командой `poetry run pre-commit install`, чтобы он указывал на новое окружение.

## 5. Какие ссылки сдавать

После merge в `main`:

- ДЗ №1: [репозиторий](https://github.com/SofiCholoian888/time_series_retail_forecasting), при необходимости ссылка на успешный запуск [Actions](https://github.com/SofiCholoian888/time_series_retail_forecasting/actions).
- ДЗ №2: [model_card.md](https://github.com/SofiCholoian888/time_series_retail_forecasting/blob/main/model_card.md) и [Model_BPMN.md](https://github.com/SofiCholoian888/time_series_retail_forecasting/blob/main/Model_BPMN.md).
- ДЗ №3: [Model_retrain_BPMN.md](https://github.com/SofiCholoian888/time_series_retail_forecasting/blob/main/Model_retrain_BPMN.md).

Для подтверждения новой оценки можно добавить небольшие `metrics.csv` и `evaluation.json` в отдельную папку `reports/runs/<run_id>/` вручную после просмотра содержимого. Полные данные и бинарные модели в Git не нужны. Не утверждайте, что целевой автоматический процесс уже развёрнут: его схема соответствует ДЗ, а интеграции явно перечислены как будущие.

## 6. Локальный запуск вместо Colab

Используйте Python 3.12 (также поддерживается 3.11), Git и [Poetry](https://python-poetry.org/docs/#installation). Poetry устанавливайте отдельно от окружения проекта, например через `pipx install poetry==2.2.1`. В терминале из корня проекта:

```bash
poetry check --lock
poetry install
poetry run pre-commit install
poetry run pre-commit run --all-files
poetry run pytest -q
poetry build
```

`poetry.toml` создаёт окружение в `.venv`. Проверить путь: `poetry env info --path`. Если Poetry выбрал неверную версию, сначала задайте `poetry env use python3.12` (в Windows — полный путь к `python.exe` версии 3.12).

После размещения `train.csv` в `data/raw/`:

```bash
poetry run retail-forecast evaluate --data data/raw/train.csv --config configs/default.json --expected-series 500 --output results/full
poetry run retail-forecast train --data data/raw/train.csv --config configs/default.json --expected-series 500 --model-dir artifacts/full-v1
poetry run retail-forecast predict --model-dir artifacts/full-v1 --output results/full/forecast.csv
poetry run python scripts/export_zip.py --output dist/retail_source_for_github.zip
```

Для повторного обучения используйте новую папку `full-v2`. Для подтверждения качества нового артефакта сначала повторите оценку на корректных временных срезах. `train` сам по себе не решает, можно ли публиковать модель.

## 7. Типичные ошибки

| Сообщение | Действие |
|---|---|
| Missing input columns / нет `sales` | Выбран неверный CSV; нужен `train.csv` |
| Expected 500 series | Загружена часть данных или другой датасет; для собственного набора осознанно измените ожидаемое число |
| Missing daily observations / aligned dates | Исправьте пропуски/разные границы рядов; код специально не подставляет нули |
| Use a new model version directory | Задайте новую папку версии или новый RUN_ID |
| ModuleNotFoundError | Запускайте через `poetry run`; проверьте завершение `poetry install` |
| pre-commit modified files | Просмотрите автоматические исправления и повторите проверку |
| В ZIP на GitHub нет просмотримых исходников | Распакуйте ZIP и загрузите файлы, как описано выше |
| В notebooks/research нет статистических результатов | Сначала полностью выполните 01 и 02; старые числа больше не подставляются |
