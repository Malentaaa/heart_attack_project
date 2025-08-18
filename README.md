# Предсказание риска сердечных приступов

Учебный проект в рамках курса по Data Science.  
Цель — разработать модель машинного обучения для предсказания вероятности сердечного приступа на основе медицинских данных.  
На текущем этапе выполнен **EDA, отбор признаков и обучение моделей**.  
В ближайшее время планируется интеграция модели в **FastAPI сервис** для использования в приложении.

---

## 📊 Датасет

- Учебный набор данных, предоставленный в рамках курса.  
- Всего **23 признака**, включая демографические данные, медицинские показатели и образ жизни.  
- Целевая переменная: `heart_attack_risk_binary` (0 — нет, 1 — риск сердечного приступа).

### Примеры признаков:
- `age` — возраст  
- `heart_rate` — пульс  
- `bmi` — индекс массы тела  
- `cholesterol`, `triglycerides`  
- `systolic_blood_pressure`, `diastolic_blood_pressure`  
- `smoking`, `alcohol_consumption`, `obesity`  
- `diabetes`, `previous_heart_problems`, `medication_use`  
- `sleep_hours_per_day`, `diet`, `income`

---

## 🔬 Методология

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

## 📈 Результаты

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

## 📂 Структура проекта  

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

## 🚀 Установка и запуск

1. Клонировать репозиторий:
   ```bash
   git clone https://github.com/Malentaaa/heart_attack_project.git
   cd <heart_attack_project>

2. Создайте виртуальное окружение и установите зависимости:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # для Linux/Mac
   .venv\Scripts\activate      # для Windows
   pip install -r requirements.txt

3. Запустите Jupyter Notebook:
   ```bash
   jupyter notebook


## 🚀 FastAPI (в разработке)

В ближайших обновлениях будет добавлен REST API для удобного взаимодействия с моделью:
- Эндпоинт для предсказаний.
- Валидация входных данных.
- Swagger UI для тестирования.

## 📦 Используемые библиотеки
- pandas, numpy — работа с данными
- matplotlib, seaborn, plotly — визуализация
- scikit-learn — препроцессинг и модели
- statsmodels — статистический анализ (VIF)
- xgboost, lightgbm, catboost — градиентный бустинг
- shap — интерпретация моделей
- fastapi — REST API (в процессе интеграции)

## 📈 Планы по развитию

 - Базовый EDA и предобработка данных
 - Обучение и сравнение моделей
 - Подбор гиперпараметров и порога классификации
 - Реализация FastAPI сервиса
 - Docker-контейнеризация проекта
 - Деплой модели на облачный сервис

## 👨‍💻 Автор
Проект выполнен Толстокоровой К.В. в рамках учебного курса по Data Science.