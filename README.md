## Мастерская Yandex Практикума: Проект "Medical QA"

### Описание проекта:
Исходные данные:
[Набор пар медицинских вопросов](https://huggingface.co/datasets/medical_questions_pairs) с метками `1` и `0` (вопросы похожи / непохожи).

Цель: Разработать рекомендательный сервис для поиска медицинских вопросов, похожих на введенный пользователем текст.

Результаты:
* Проверили исходный датасет на отсутствие пропусков, дублирующих пар вопросов и сбалансированность меток;
* Создали корпус вопросов и добавили их id в исходный датасет;
* Применили несколько способов предобработки и векторизации текста, наибольшая точность поиска `accuracy@n` получилась у языковой модели BERT, предобученной на определении схожести текстов (SentenceTransformer);
* Обучили два классификатора (CatBoostClassifier и BERT с добавленным линейным слоем), чтобы из вопросов, ближайших к заданному по векторному расстоянию, отбирать те, для которых классификатор выдает 1. Обучение проводилось на расширенном датасете, содержащим исходные пары вопросов и пары с вопросами в обратном порядке. Наибольшая точность классификации на тестовой выборке получилась у BERT (0.90 против 0.77 у catboost);
* Создали микросервис и клиентскую часть на базе FastAPI и Streamlit соответственно.
  В пользовательском веб-интерфейсе можно выбрать тип классификатора либо отказаться от него для ускорения работы;
  
Выполнил:
Кирилл Носов, email: ibnkir@yandex.ru, tg: [Xmg23](https://t.me/Xmg23).

### Структура репозитория:
* data - локально сохраненные данные;
* models - сериализованные модели обученных классификаторов;
* notebooks - тетрадки Jupyter Notebooks для EDA, прототипирования решения и обучения классификаторов;
* app - код FastAPI и Streamlit приложений.

### Как воспользоваться репозиторием:
0. Убедитесь, что на вашем компьютере установлен Python и все необходимые пакеты (см. файл requirements.txt).
1. Перейти в домашнюю папку и склонировать репозиторий на ваш компьютер
   ```bash
   cd ~
   git clone https://github.com/ibnkir/medical_qa_project
   ```
2. Скачать обученные классификаторы [catboost](https://disk.yandex.ru/d/lAHRjzjzJWSGTw) 
и/или [fine-tuned BERT](https://disk.yandex.ru/d/4StkjpA41bL0oA) по указанным ссылкам 
и сохранить их в подпапку `models/` репозитория (этот пункт можно пропустить, тогда при отправке запросов в выпадающем списке выбора классификатора указать None).
     
3. Открыть терминал, перейти в подпапку `app/` и запустить серверную часть 
(если порт 8081 уже занят, то заменить его на другой)
   ```bash
   uvicorn fastapi_app:app --reload --port 8081 --host 0.0.0.0
   ```
4. Открыть другой терминал, перейти в подпапку `app/` и запустить клиентскую часть
   ```bash
   streamlit run streamlit_app.py
   ```
