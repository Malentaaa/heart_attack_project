# Предсказание риска сердечных приступов

Учебный проект в рамках курса по Data Science.  
Цель — разработать модель машинного обучения для предсказания вероятности сердечного приступа на основе медицинских данных.  
На текущем этапе выполнен **EDA, отбор признаков и обучение моделей**.  
В ближайшее время планируется интеграция модели в **FastAPI сервис** для использования в приложении.

---

📊 Датасет

- Учебный набор данных, предоставленный в рамках курса.  
- Всего **23 признака**, включая демографические данные, медицинские показатели и образ жизни.  
- Целевая переменная: `heart_attack_risk_binary` (0 — нет, 1 — риск сердечного приступа).

Примеры признаков:
- `age` — возраст  
- `heart_rate` — пульс  
- `bmi` — индекс массы тела  
- `cholesterol`, `triglycerides`  
- `systolic_blood_pressure`, `diastolic_blood_pressure`  
- `smoking`, `alcohol_consumption`, `obesity`  
- `diabetes`, `previous_heart_problems`, `medication_use`  
- `sleep_hours_per_day`, `diet`, `income`

---

🔬 Методология

Этапы анализа:

1. **Предобработка данных**
   - Проверка пропусков и типов признаков  
   - Импьютация (SimpleImputer)  
   - Масштабирование (StandardScaler, MinMaxScaler)  
   - Кодирование категорий (OneHotEncoder, OrdinalEncoder)  

2. **Разведочный анализ (EDA)**
   - Статистическое описание  
   - Проверка нормальности (Shapiro-Wilk)  
   - Корреляции и мультиколлинеарность (VIF)  
   - Визуализации (гистограммы, тепловые карты, boxplot)

3. **Отбор признаков**
   - `SelectKBest` (f_classif, mutual_info_classif)  
   - Из 23 признаков отобрано 15 наиболее информативных:  
     `heart_rate, income, bmi, triglycerides, sleep_hours_per_day, systolic_blood_pressure, diastolic_blood_pressure, diet, diabetes, family_history, smoking, obesity, alcohol_consumption, previous_heart_problems, medication_use`

4. **Моделирование**
   - Random Forest 🌟 (лучшая модель)  
   - Gradient Boosting  
   - XGBoost, LightGBM, CatBoost  
   - SVM  

5. **Оценка качества**
   - Метрики: F2, F1, Recall, Precision, ROC-AUC, PR-AUC  
   - Подбор порога классификации

6. **Интерпретация**
   - Feature Importance  
   - SHAP values

---

📈 Результаты

**Лучшая модель**: Random Forest (с подбором порога по F2).  

| Метрика        | Значение |
|----------------|----------|
| F2             | 0.7338   |
| F1             | 0.5243   |
| Recall         | 1.0      |
| Precision      | 0.3553   |
| ROC-AUC        | 0.558    |
| PR-AUC         | 0.4156   |
| BestThreshold  | 0.39     |

➡️ **Recall = 1.0** — модель «ловит» все случаи (важно для медицины), но Precision пока невысокий.  
Задача следующего этапа — улучшить баланс чувствительности и точности.

---

📂 Структура проекта  

project_heart_attack/
├── notebooks/ # Jupyter ноутбуки (EDA, моделирование)
│ └── project_heart_attack.ipynb
├── src/ # Python-модули (препроцессинг, пайплайны, модели)
├── data/
│ ├── raw/ # Сырые данные (не загружаются в Git)
│ ├── processed/ # Обработанные данные
│ └── results/ # Графики и метрики
├── models/ # Сохранённые модели (.pkl)
├── images/ # Визуализации для отчёта/README
├── requirements.txt # Зависимости проекта
└── README.md # Описание проекта

🚀 Установка и запуск

1. Клонировать репозиторий:
   ```bash
   git clone https://github.com/Malentaaa/heart_attack_project.git
   cd heart_attack_project>

2. Создайте виртуальное окружение и установите зависимости:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # для Linux/Mac
   .venv\Scripts\activate      # для Windows
   pip install -r requirements.txt

3. Запустите Jupyter Notebook:
   ```bash
   jupyter notebook


🚀 FastAPI-сервис (готов)

В репозитории есть рабочий REST-API и простая веб-страница для загрузки CSV и получения предсказаний.
Что умеет
📄 Веб-страница GET / для загрузки CSV.
🔍 Эндпоинт POST /api/predict_csv — пакетные предсказания из CSV.
🔹 Эндпоинт POST /api/predict — предсказание по одному объекту (JSON).
🧪 Swagger UI GET /docs — удобно тестировать API.
🩺 GET /api/model_status — проверка, что модель загрузилась (порог, шаги пайплайна, ожидаемые фичи).
❤️ GET /health — простой health-check.

⚙️ Установка и запуск
1) Клонировать и войти в проект
git clone https://github.com/Malentaaa/heart_attack_project.git
cd project_heart_attack

2) Виртуальное окружение и зависимости
python -m venv .venv
indows PowerShell:
.\.venv\Scripts\Activate.ps1
MacOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

3) Убедиться, что артефакты модели на месте
models/heart_rf_final.pkl — конвейер sklearn (prep → select → clf) + threshold
models/expected_features.json — список "сырых" признаков (порядок колонок)
models/threshold.json (необязательно) — если есть, переопределяет порог инференса

4) Запуск API
Важно: запускаем из корня проекта
set PYTHONPATH=.
uvicorn app_backend.app:app --reload --port 8001


Открой:

Веб-страница: http://127.0.0.1:8001/

Swagger: http://127.0.0.1:8001/docs

Статус модели: http://127.0.0.1:8001/api/model_status

📁 Структура бэкенда
app_backend/
├─ app.py                # точка входа FastAPI (раздаёт / и /static, подключает роуты)
├─ __init__.py
├─ static/
│  └─ index.html         # простая страница загрузки CSV
├─ routers/
│  ├─ __init__.py
│  └─ predict.py         # эндпоинты API
└─ deps/
   ├─ __init__.py
   └─ model.py           # загрузка модели, threshold, expected_features


Артефакты модели:

models/
├─ heart_rf_final.pkl          # {"pipeline": sklearn.Pipeline, "threshold": float}
├─ expected_features.json      # список "сырых" колонок, которых ждёт препроцессор
└─ threshold.json (опц.)       # {"threshold": 0.39} — имеет приоритет над pkl

🔌 Эндпоинты
GET /

Отдаёт страницу загрузки CSV. Колонки автоматически нормализуются:
заголовки приводятся к snake_case (пробелы/капс → подчёркивания/нижний регистр);
Gender нормализуется в {0,1};
служебные колонки типа Unnamed: 0 игнорируются;
недостающие признаки добиваются NaN и заполняются имьютерами внутри preprocessor.
GET /api/model_status

Пример ответа:

{
  "ok": true,
  "has_predict_proba": true,
  "threshold": 0.39,
  "expected_features": ["age", "cholesterol", "..."],
  "pipeline_steps": [
    {"name": "prep", "type": "ColumnTransformer"},
    {"name": "select", "type": "SelectKBest"},
    {"name": "clf", "type": "RandomForestClassifier"}
  ]
}

POST /api/predict_csv (multipart/form-data)

file: CSV-файл.
Ответ:

{
  "total": 966,
  "positive": 123,
  "negative": 843,
  "items": [
    {"row_id": 1, "label": 0, "proba": 0.12},
    {"row_id": 2, "label": 1, "proba": 0.81}
  ]
}

POST /api/predict (application/json)

Пример запроса:

{
  "age": 58,
  "cholesterol": 220,
  "heart_rate": 150,
  "diabetes": 0,
  "gender": "male",
  "systolic_blood_pressure": 140,
  "diastolic_blood_pressure": 90
}


Необязательные поля можно опустить — их обработает препроцессор.

🧠 Про модель

Экспорт выполняется из ноутбука: конвейер Pipeline(prep → select → clf) сохраняется вместе с порогом классификации.

Минимальные требования к артефактам:
heart_rf_final.pkl — содержит хотя бы ключ "pipeline" со sklearn-Pipeline; опционально "threshold".
expected_features.json — список колонок сырого входа (помогает выровнять CSV перед инференсом).
threshold.json (опционально) — если присутствует, имеет приоритет над порогом из .pkl.

🧪 Примеры (curl)
одно наблюдение
curl -X POST http://127.0.0.1:8001/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"age\":58, \"cholesterol\":220, \"gender\":\"male\"}"

CSV
curl -X POST http://127.0.0.1:8001/api/predict_csv \
  -F "file=@data/processed/heart_test.csv"

🛠️ Частые проблемы и решения

Страница не открывается / 404 на “/”
Запускать именно uvicorn app_backend.app:app --reload из корня проекта.
Проверить, что app_backend/static/index.html существует.

Could not import module "app" / конфликт имён
Папка проекта не должна называться fastapi, иначе конфликтует с библиотекой. Мы используем app_backend.

NaN и SelectKBest ругается
Проверь /api/model_status: порядок шагов должен быть prep → select → clf.
Если в старом .pkl был порядок prep → clf → select, пересохраните модель или используйте наш загрузчик — он переставит шаги автоматически.

Порог не тот
Обновите models/threshold.json:

{"threshold": 0.39}

📦 Используемые библиотеки

pandas, numpy — работа с данными
matplotlib, seaborn, plotly — визуализация
scikit-learn — препроцессинг и модели
statsmodels — статистический анализ (VIF)
xgboost, lightgbm, catboost — градиентный бустинг
shap — интерпретация моделей
fastapi, uvicorn — REST API

📈 Планы по развитию

Базовый EDA и предобработка данных
Обучение и сравнение моделей
Подбор гиперпараметров и порога классификации
Реализация FastAPI-сервиса (страница + REST API)
Docker-контейнеризация проекта
Деплой модели на облачный сервис

👨‍💻 Автор
Проект выполнен Толстокоровой К.В. в рамках учебного курса по Data Science.