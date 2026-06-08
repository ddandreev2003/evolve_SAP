# Запуск эволюции SAP (OpenEvolve)

Краткое руководство: что настраивать, где менять параметры и как запускать обучение системного промпта для SAP.

---

## Что делает эволюция

OpenEvolve **не обучает FLUX**. Он мутирует Python-файл с переменной `SYSTEM_PROMPT` — инструкциями для Qwen, как декомпозировать противоречивые промпты по технике **SAP** (Stage-Aware Prompting).

На каждом успешном цикле:

1. **Gemini** (`google/gemini-3.1-pro-preview`) предлагает правки в `SYSTEM_PROMPT`
2. **Qwen** (`qwen/qwen3.5-35b-a3b`) декомпозирует 3 тестовых промпта
3. **FLUX** (локально) генерирует картинки с переключением промптов на шагах деноising
4. **VL-модель** + **Gemini-judge** считают `combined_score`

```
combined_score = 0.8 × (alignment / 5) + 0.2 × (gemma / 5)
```

Цель — максимизировать `combined_score`.

---

## Подготовка окружения

### 1. Активировать venv

```bash
source /home/ubuntu/venv/bin/activate
cd /home/ubuntu/evolve_SAP
```

### 2. Проверить зависимости

```bash
python -c "import torch, diffusers, openevolve, openai; print('OK')"
which python   # ожидается /home/ubuntu/venv/bin/python
```

### 3. Файл `.env` (основные секреты и пути)

Создайте или отредактируйте `.env` в корне проекта:

```bash
# RouterAI — все облачные LLM (Qwen, VL, Gemini)
ROUTERAI_API_KEY=YOUR_API_KEY
ROUTERAI_BASE_URL=https://routerai.ru/api/v1

# Локальная модель FLUX (обязательно)
SAP_FLUX_MODEL_PATH=/absolute/path/to/FLUX.2-klein-base-4B

# Параметры eval во время эволюции
SAP_NUM_INFERENCE_STEPS=50
SAP_SEEDS_LIST=30498,30499
SAP_IMAGE_HEIGHT=512
SAP_IMAGE_WIDTH=512
```

Перед запуском подгрузите переменные:

```bash
set -a && source .env && set +a
```

> `.env` в `.gitignore` — не коммитьте ключи.

### 4. FLUX на диске

Путь `SAP_FLUX_MODEL_PATH` должен указывать на локальную папку с весами `FLUX.2-klein-base-4B`. Скачивание из Hub в strict-режиме эволюции не используется.

---

## Где менять конфигурацию

Конфигурация разбита на **четыре уровня** — от глобальных путей до логики мутаций.

| Уровень | Файл | Что меняется |
|---------|------|--------------|
| **Секреты и eval** | `.env` | API-ключ, путь к FLUX, шаги, сиды, размер картинки |
| **OpenEvolve** | `openevolve_sap/configs/multi_gpu.yaml` | LLM для мутаций, число GPU-воркеров, чекпоинты, популяция |
| **Мета-промпт эволюции** | `openevolve_sap/prompts/evolution_system_message.md` | Инструкции для Gemini: цель SAP, стратегии A/B, формат мутаций |
| **Начальный кандидат** | `openevolve_sap/initial_program.py` | Стартовый `SYSTEM_PROMPT` (поколение 0) |
| **Тестовые промпты** | `openevolve_sap/prompt_set.json` | 3 противоречивых промпта для оценки fitness |

CLI-флаги (`--iterations`, `--gpus` и т.д.) **перекрывают** часть YAML на время одного запуска.

---

## `.env` — переменные eval и производительности

| Переменная | По умолчанию | Назначение |
|------------|--------------|------------|
| `ROUTERAI_API_KEY` | — | **Обязательно.** Ключ RouterAI |
| `SAP_FLUX_MODEL_PATH` | — | **Обязательно.** Локальный FLUX |
| `SAP_NUM_INFERENCE_STEPS` | `30` | Шаги деноising в eval (в `.env` часто `50`) |
| `SAP_SEEDS_LIST` | `30498,30499` | Сиды через запятую; по одной картинке на сид |
| `SAP_IMAGE_HEIGHT` / `SAP_IMAGE_WIDTH` | `512` | Размер генерируемых картинок |
| `SAP_RAM_LIMIT_GB` | 75% RAM | Лимит RSS на процесс |
| `SAP_BATCH_SAP` | `1` | Батч-декомпозиция всех 3 промптов одним вызовом Qwen |
| `SAP_EVAL_PIPELINE` | `1` | VL-счёт prompt *i* параллельно с FLUX prompt *i+1* |
| `SAP_VL_MAX_CONCURRENT` | `3` | Параллельные VL-запросы (если pipeline выключен) |
| `SAP_CLEANUP_EVERY_N_PROMPTS` | `0` | `0` = gc/cuda после каждого промпта; `3` = реже |
| `SAP_CASCADE_EVAL` | off | `1` = быстрый stage1 (1 промпт) перед полным eval |
| `SAP_CASCADE_THRESHOLDS` | — | Пороги cascade, напр. `0.35,0.0` |
| `SAP_MAX_EVOLUTION_ATTEMPTS` | `iterations × 8` | Потолок попыток при ретраях неудачных циклов |

**Оценка времени одного полного eval** (3 промпта × N сидов × M шагов):

```
время_на_120_циклов ≈ (120 / num_gpus) × T_one_eval
```

Пример: 4 GPU, ~7 мин/eval → **~3.5–4 часа** на 120 успешных циклов при 50 шагах и 2 сидах.

Чтобы ускорить (больше циклов в час, ниже качество eval):

```bash
SAP_NUM_INFERENCE_STEPS=30
SAP_SEEDS_LIST=30498
```

---

## YAML — `openevolve_sap/configs/multi_gpu.yaml`

Основной конфиг для 4× GPU. Ключевые поля:

```yaml
max_iterations: 80          # дефолт в файле; обычно переопределяется --iterations
checkpoint_interval: 50     # или --checkpoint-interval

llm:
  primary_model: "google/gemini-3.1-pro-preview"   # модель мутаций
  api_base: "https://routerai.ru/api/v1"
  temperature: 0.7
  max_tokens: 8000
  timeout: 120

database:
  population_size: 40
  archive_size: 20
  num_islands: 1            # один остров — проще планирование на SAP eval
  elite_selection_ratio: 0.2
  exploitation_ratio: 0.7

evaluator:
  timeout: 3600             # секунд на один eval (FLUX + API)
  parallel_evaluations: 4   # = число GPU; scheduler выставляет len(--gpus)

diff_based_evolution: true  # Gemini отдаёт diff, не полный rewrite
max_tasks_per_child: 30     # перезапуск worker после N eval (защита от утечек RAM)

early_stopping_patience: null   # null = не останавливаться раньше --iterations
```

### Что менять чаще всего

| Задача | Где |
|--------|-----|
| Число циклов эволюции | CLI `--iterations 120` (не только `max_iterations` в YAML) |
| Частота чекпоинтов | CLI `--checkpoint-interval 5` |
| Модель мутаций | `llm.primary_model` в YAML |
| Число GPU | CLI `--gpus 0 1 2 3` |
| Ранний стоп по плато | `early_stopping_patience: 20` (сейчас отключено) |
| Быстрый отсев слабых кандидатов | `cascade_evaluation: true` + `SAP_CASCADE_EVAL=1` |

Для одиночного GPU используйте `openevolve_sap/config.yaml` (`parallel_evaluations: 1`).

---

## Мета-промпт эволюции

Файл: `openevolve_sap/prompts/evolution_system_message.md`

Загружается **в рантайме** в `scheduler.py` и подставляется как `system_message` для Gemini при мутациях. Содержимое YAML (`prompt.system_message`) игнорируется — там только заглушка.

Редактируйте этот файл, если нужно:

- явнее объяснить цель (написать хороший `SYSTEM_PROMPT` для SAP);
- добавить anti-patterns и примеры Strategy A / B;
- уточнить формат вывода для Qwen (`prompts_list`, `switch_prompts_steps`).

После правки **перезапустите** эволюцию — hot-reload нет.

---

## Начальный промпт и тестовый набор

### `openevolve_sap/initial_program.py`

Стартовый кандидат с `SYSTEM_PROMPT = """..."""`. От него начинается популяция.

### `openevolve_sap/prompt_set.json`

Фиксированные промпты для fitness (сейчас 3 штуки):

```json
[
  "A bouquet of flowers is upside down in a vase",
  "A white glove has 6 fingers",
  "The shadow of a cat is facing the opposite direction"
]
```

Добавление промптов **увеличивает время каждого eval** пропорционально их числу.

---

## Запуск

### Базовая команда (foreground)

```bash
source /home/ubuntu/venv/bin/activate
cd /home/ubuntu/evolve_SAP
set -a && source .env && set +a

python scripts/run_evolution.py \
  --config openevolve_sap/configs/multi_gpu.yaml \
  --iterations 120 \
  --checkpoint-interval 5 \
  --gpus 0 1 2 3 \
  --log-level INFO
```

### Фоновый запуск (переживает disconnect SSH, если VM не выключена)

```bash
nohup /home/ubuntu/venv/bin/python scripts/run_evolution.py \
  --config openevolve_sap/configs/multi_gpu.yaml \
  --iterations 120 \
  --checkpoint-interval 5 \
  --gpus 0 1 2 3 \
  > openevolve_sap/experiments/evolution_run.log 2>&1 &

echo $!   # PID процесса
```

### Smoke test (2–4 цикла)

```bash
python scripts/run_evolution.py \
  --iterations 4 \
  --gpus 0 1 2 3 \
  --checkpoint-interval 2
```

### Все CLI-флаги

| Флаг | Описание |
|------|----------|
| `--config` | YAML-конфиг (default: `openevolve_sap/configs/multi_gpu.yaml`) |
| `--gpus 0 1 2 3` | Физические индексы GPU |
| `--iterations` / `-i` | **Число успешных eval-циклов** (см. ниже) |
| `--checkpoint-interval` | Сохранять checkpoint каждые N успешных циклов |
| `--checkpoint PATH` | Продолжить с `checkpoint_<N>` |
| `--experiment-dir PATH` | Фиксированная папка эксперимента (иначе `experiment_<timestamp>`) |
| `--output` / `-o` | Куда писать OpenEvolve output (default: `openevolve_sap/output`) |
| `--ram-limit-gb` | Лимит RAM на процесс |
| `--target-score` | Остановиться при достижении `combined_score` |
| `--export-best` | Путь для экспорта лучшего текста (default: `openevolve_sap/best_evolved_system_prompt.txt`) |

Алиас: `python openevolve_sap/run_openevolve_sap.py` — то же самое.

---

## 120 циклов = 120 успешных eval

С патчем `openevolve_sap/core/evolution_patch.py`:

- в зачёт идут только циклы с **полным eval** (mutation + FLUX + score);
- ошибки Gemini («No valid diffs»), таймауты и падения worker **не считаются** — цикл повторяется;
- при падении process pool — **автоматический restart**;
- in-flight eval **дожидаются** перед остановкой.

В логе ищите:

```
SAP evolution patch installed
SAP: evolution target = 120 successful eval cycles
Iteration 47: Program ... [successful 47/120]
✅ Evolution completed - 120/120 successful eval cycles finished
```

---

## Продолжение с чекпоинта

```bash
set -a && source .env && set +a

python scripts/run_evolution.py \
  --checkpoint openevolve_sap/output/checkpoints/checkpoint_50 \
  --experiment-dir openevolve_sap/experiments/experiment_YYYYMMDD_HHMMSS \
  --iterations 70 \
  --gpus 0 1 2 3 \
  --checkpoint-interval 5
```

`--iterations` при resume — сколько **ещё** успешных циклов выполнить.

---

## Мониторинг

### Логи

| Путь | Содержимое |
|------|------------|
| `openevolve_sap/experiments/evolution_run.log` | stdout/stderr при `nohup` |
| `openevolve_sap/experiments/experiment_<ts>/experiment.jsonl` | Структурированный лог |
| `openevolve_sap/experiments/experiment_<ts>/gpu_metrics.csv` | GPU каждые 5 с |
| `openevolve_sap/experiments/experiment_<ts>/evolution_stats.csv` | best/avg score по чекпоинтам |

```bash
tail -f openevolve_sap/experiments/evolution_run.log
grep "successful" openevolve_sap/experiments/evolution_run.log | tail -20
```

### GPU

```bash
watch -n 2 nvidia-smi
```

### Проверить, что процесс жив

```bash
pgrep -af "run_evolution.py"
```

---

## Визуализация (во время или после прогона)

```bash
source /home/ubuntu/venv/bin/activate
cd /home/ubuntu/evolve_SAP
export PYTHONPATH=.

python openevolve_sap/visualization/visualizer.py \
  --checkpoint openevolve_sap/output/checkpoints/checkpoint_120 \
  --experiment-dir openevolve_sap/experiments/experiment_YYYYMMDD_HHMMSS \
  --host 0.0.0.0 --port 8050
```

Открыть: `http://127.0.0.1:8050` (или IP VM с портом 8050).

Вкладки: **Branching** (дерево программ), **Performance** (GPU), **List**, **Evals** (галерея `eval_results/`).

Перезапуск только UI (эволюцию не останавливает):

```bash
pkill -f "openevolve_sap/visualization/visualizer.py"
# затем снова python visualizer.py ...
```

---

## Результаты

| Путь | Описание |
|------|----------|
| `openevolve_sap/best_evolved_system_prompt.txt` | Лучший текст `SYSTEM_PROMPT` |
| `openevolve_sap/output/best/best_program.py` | Лучший Python-кандидат |
| `openevolve_sap/output/checkpoints/checkpoint_<N>/` | Снимок БД OpenEvolve + `best_program.py` |
| `openevolve_sap/experiments/.../eval_results/<run_id>/` | Картинки, decomposition, scores |

Render-only (без API-судей):

```bash
python openevolve_sap/evaluator.py --program openevolve_sap/output/best/best_program.py
```

---

## Модели по ролям

| Роль | Модель | Где задаётся |
|------|--------|--------------|
| Мутации (эволюция) | `google/gemini-3.1-pro-preview` | `multi_gpu.yaml` → `llm.primary_model` |
| Мета-промпт для мутаций | текст в `evolution_system_message.md` | scheduler |
| SAP-декомпозиция | `qwen/qwen3.5-35b-a3b` | `llm_interface/llm_SAP.py` |
| VL alignment | `qwen/qwen3-vl-235b-a22b-thinking` | `benchmarks/gpt_eval.py` |
| Judge качества промпта | `google/gemini-3.1-pro-preview` | `openevolve_sap/evaluator.py` |
| Генерация картинок | `FLUX.2-klein-base-4B` (локально) | `SAP_FLUX_MODEL_PATH` |

---

## Типичные проблемы

| Симптом | Решение |
|---------|---------|
| `SAP_FLUX_MODEL_PATH must be set` | Заполнить `.env` и `source .env` |
| `ROUTERAI_API_KEY is not set` | Ключ в `.env` |
| `Need 4 GPUs, found N` | Уменьшить `--gpus` или освободить GPU |
| Мало eval за время прогона | Уменьшить `SAP_NUM_INFERENCE_STEPS`, сиды; больше GPU |
| `process pool terminated abruptly` | Патч перезапустит pool; проверить RAM/OOM в `dmesg` |
| Визуализатор показывает неполные runs | Папки без `manifest.json` — оборванные eval; смотреть только с manifest |

---

## Быстрый чеклист перед запуском

- [ ] `source /home/ubuntu/venv/bin/activate`
- [ ] `.env` с `ROUTERAI_API_KEY` и `SAP_FLUX_MODEL_PATH`
- [ ] `nvidia-smi` — нужные GPU свободны
- [ ] Выбраны `--iterations` и `--checkpoint-interval`
- [ ] При необходимости отредактированы `evolution_system_message.md` и `initial_program.py`
- [ ] Для длинного прогона — `nohup` + `tail -f` на лог

Подробнее об архитектуре — в [README.md](README.md).
