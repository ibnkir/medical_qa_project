## Проект "Medical QA"

### Описание проекта:
Исходные данные:
[Набор пар медицинских вопросов с сайта Hugging Face](https://huggingface.co/datasets/medical_questions_pairs) с метками `1` и `0` (вопросы похожи / непохожи).

Цель: Разработать рекомендательный микросервис на базе FastAPI для поиска медицинских вопросов, похожих на введенный пользователем текст.

Основные инструменты:
- Visual Studio Code,
- Scikit-learn,
- Catboost,
- Transformers,
- FastAPI, 
- Uvicorn,
- Streamlit.

Результаты:
* Проверили исходный датасет на отсутствие пропусков, дублирующих пар вопросов и сбалансированность меток;
* Создали корпус вопросов и добавили их id в исходный датасет;
* Применили несколько способов предобработки и векторизации текста, включая BoW, TF-IDF, самостоятельно обученную модель word2vec и предобученные glove и BERT. Наибольшая точность поиска `accuracy@n` 
(рассчитывается, как доля случайно отобранных вопросов, у которых среди найденных n ближайших соседей есть такие, что соответствующая пара в исходном датасете имеет метку `1`)
получилась у языковой модели BERT, предобученной на определении схожести текстов (SentenceTransformer);
* Обучили классификаторы CatBoostClassifier и fine-tuned BERT с добавленным линейным слоем, чтобы из вопросов, ближайших к заданному по векторному расстоянию, отбирать те, для которых классификатор выдает `1`. Обучение проводилось на расширенном датасете, содержащим исходные пары вопросов и пары с вопросами в обратном порядке. Наибольшая точность классификации на тестовой выборке получилась у BERT (0.90 против 0.77 у catboost);
* Создали микросервис и клиентскую часть на базе FastAPI и Streamlit соответственно.
  В пользовательском веб-интерфейсе можно вводить свой текст, задавать требуемое количество похожих вопросов и выбирать тип классификатора.
  
Планируемые доработки:
* Вместо добавления к BERTу слоя классификатора и дообучения его на кросс-энтропии использовать в качестве функции потерь triplet-loss и дообучать сами эмбединги.

Выполнил:
Кирилл Носов, email: ibnkir@yandex.ru, tg: [Xmg23](https://t.me/Xmg23).

### Структура репозитория:
* `data/` - папка с локально сохраненными данными;
* `models/` - папка с сериализованными моделями обученных классификаторов;
* `notebooks/` - папка с тетрадкой Jupyter Notebooks для EDA, прототипирования решения и обучения классификаторов;
* `app/` - папка с исходным кодом FastAPI- и Streamlit-приложений.

### Как воспользоваться репозиторием:
0. Убедитесь, что на вашем компьютере установлен Python, Git и все необходимые пакеты (см. файл requirements.txt).
1. Перейти в домашнюю папку и склонировать репозиторий на ваш компьютер
   ```bash
   cd ~
   git clone https://github.com/ibnkir/medical_qa_project
   ```
2. Скачать обученные модели [catboost](https://disk.yandex.ru/d/lAHRjzjzJWSGTw) 
и/или [fine-tuned BERT](https://disk.yandex.ru/d/4StkjpA41bL0oA) по указанным ссылкам 
и сохранить их в папке `models/` репозитория (этот пункт можно пропустить, тогда поиск похожих вопросов будет проводиться только по векторному расстоянию без использования классификатора).
     
3. Открыть терминал, перейти в папку `app/` и запустить серверную часть 
(если порт 8081 уже занят, то заменить его на другой)
   ```bash
   uvicorn fastapi_app:app --reload --port 8081 --host 0.0.0.0
   ```
4. Открыть другой терминал, перейти в папку `app/` и запустить клиентскую часть
   ```bash
   streamlit run streamlit_app.py
   ```
