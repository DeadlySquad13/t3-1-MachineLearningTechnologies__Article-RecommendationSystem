# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Рекомендательные системы
# Выполнил работу **Пакало Александр Сергеевич**, студент РТ5-61Б


# %% [markdown]
# # Цель работы
# Необходимо предложить пользователям Яндекс.Карт соответствующие их вкусу
# кафе, бары и рестораны в неродном городе: москвичам – в Санкт-Петербурге, а
# петербуржцам – в Москве. В качестве данных предоставлена анонимизированная
# информация о реальных отзывах и оценках, оставляемых пользователями
# Яндекс.Карт на заведения общепита Москвы и Санкт-Петербурга, и различная
# информация о самих заведениях.

# В частности, каждый отзыв содержит множество
# аспектов (упомянутые в отзыве блюда, особенности и т. п.), извлеченных из
# отзыва с помощью NLP-алгоритма. Для заданного множества москвичей и
# петербуржцев нужно предсказать, какие заведения в неродном городе они
# посетят, оставив при этом положительный отзыв с оценкой 4 или 5.

# Так как данных очень много, перед тем как приступить к анализу, проведем обзор данных и, возможно, потребуется их предобработка, чтобы датасет стал более удобным и пригодным к проведению исследования.
# 
# Таким образом исследование пройдет в 6 этапов:
# - загрузка данных,
# - проведение разведочного анализа данных и предобработка данных,
# - разделение на обучающую и тестовую выборку,
# - выбор метрики,
# - подбор рекомендаций с помощью различных моделей,
# - анализ результатов моделей на основе метрики.

# %% [markdown]
# # Импорт библиотек

# %%
# Основные библиотеки.
import numpy as np
import pandas as pd

# Визуализция.
import matplotlib.pyplot as plt
import seaborn as sns

# Для матрицы взаимодействий.
from scipy import sparse

# Отрисовка статуса выполнения.
from tqdm.notebook import tqdm

# Типизация.
from typing import List

# Настройки отрисовки графиков.
# %matplotlib inline

# %% [markdown]
# # Загрузка данных

# %% [markdown]
# ## Считываем данные из .csv
# Загрузим файлы датасета в помощью библиотеки Pandas.
# 
# Не смотря на то, что файлы имеют расширение txt они представляют собой данные
# в формате [CSV](https://ru.wikipedia.org/wiki/CSV). Часто в файлах такого
# формата в качестве разделителей используются символы ",", ";" или табуляция.
# Поэтому вызывая метод read_csv всегда стоит явно указывать разделитель данных
# с помощью параметра sep. Чтобы узнать какой разделитель используется в файле
# его рекомендуется предварительно посмотреть в любом текстовом редакторе.

# %% [markdown]
# В первую очередь нам понадобятся данные по **пользователям**, **организациям** и сами **отзывы**. 

# %%
users = pd.read_csv('data/users.csv', sep=',')

# %%
orgs = pd.read_csv('data/organisations.csv', sep=',')

# %%
# Читаем датасет по частям, затем соединяем в один. Позволяет загружать большие
#   (500 мб и более) датасеты.
#   К сожалению, даже при low_memory=True иногда падает кернел, несмотря на то
#   что внутри используется схожий механизм.
# - Количество рядов, содеражихся в одной части датасета.
chunksize = 1000
reviews = pd.concat(pd.read_csv('data/reviews.csv', chunksize=chunksize))

# %% [markdown]
# # Проведение разведочного анализа данных. Построение графиков, необходимых для понимания структуры данных. Анализ и предобработка данных.

# %% [markdown]
# Размеры датасетов:
# - Пользователи - 1.250.944 строк, 2 колонки.
# - Организации - 66.346 строк, 6 колонок.
# - Отзывы - 3.642.383 строк, 5 колонок.

# %%
users.shape, orgs.shape, reviews.shape

# %% [markdown]
# Общий вид данных таблицы пользователей:

# %%
users.head()

# %% [markdown]
# Список колонок:

# %%
users.columns

# %% [markdown]
# Список колонок с типами данных:

# %%
users.dtypes

# %% [markdown]
# Общий вид данных таблицы организаций:

# %%
orgs.head()

# %% [markdown]
# Список колонок:

# %%
orgs.columns

# %% [markdown]
# Список колонок с типами данных:

# %%
orgs.dtypes

# %% [markdown]
# Общий вид данных таблицы рекомендаций:

# %%
reviews.head()

# %% [markdown]
# Список колонок:

# %%
reviews.columns

# %% [markdown]
# Список колонок с типами данных:

# %%
reviews.dtypes


# %% [markdown]
# Некоторые данные (такие как рубрики и признаки), представлены строками
# значений. Для удобства анализа преобразуем их в списки чисел. 

# %%
# Переводит строку вида '1 2 3 4' в список [1, 2, 3, 4].
to_int_list = lambda values: [int(value) for value in str(values).split(' ')]

def apply_to_columns(df: pd.DataFrame, columns: List[str], func=to_int_list):
    """
    apply_to_columns(df: pd.DataFrame, columns: List[str], func=to_int_list)
        Применяет функцию к заданным колонкам <columns> датасета <df>.
    """

    for column in columns:
        df.loc[~df[column].isnull(), column] = df.loc[
            ~df[column].isnull(), column
        ].apply(func)

# %%
# Переводим рубрики и признаки в удобный вид.
columns = ['rubrics_id', 'features_id']
apply_to_columns(orgs, columns)

orgs.head()


# %% [markdown]
# Чтобы не делать __join__ каждый раз, когда нам потребуется узнать, из какого
# города организация или пользователь, сразу добавим эту информацию в отзывы.

# %%
# Join по user_id.
reviews = reviews.merge(users, on='user_id')
reviews = reviews.rename({'city': 'user_city'}, axis=1)

# Join по org_id.
reviews = reviews.merge(orgs[['org_id', 'city']], on='org_id')
reviews = reviews.rename({'city': 'org_city'}, axis=1)

# В колонке aspects тоже находятся записи вида '1 2 3', приведём их к числовому
#   списку.
columns = ['aspects']
apply_to_columns(reviews, columns)

reviews.head()

# %% [markdown]
# Отлично, теперь с отзывами будет удобно работать. 

# %% [markdown]
# # # Построение графиков для понимания структуры данных
# Посмотрим на распределение новых отзывов по дням, чтобы понять, как лучше организовать валидацию. 

# %%
sns.displot(data=reviews, x='ts', height=8)
plt.title('Распределение отзывов по дням')
plt.show()

# %% [markdown]
# Всего в выборку попали отызывы за **1216** дней. 

# %%
reviews['ts'].max()

# %% [markdown]
# # Выделение обучающей и тестовой выборки
# В качестве обучающей выборки можно взять некоторый процент от всех данных,
# однако намного эффективнее и более приближено к реальности разделение по
# времени [1, стр. 232].


# %% [markdown]
# Отложим в тестовую выборку отзывы за последние **100** дней.


# %%
def clear_df(df, suffixes=['_x', '_y'], inplace=True):
    '''
    clear_df(df, suffixes=['_x', '_y'], inplace=True)
        Удаляет из входного df все колонки, оканчивающиеся на заданные суффиксы. 
        
        Parameters
        ----------
        df : pandas.DataFrame
        
        suffixies : Iterable, default=['_x', '_y']
            Суффиксы колонок, подлежащих удалению
            
        inplace : bool, default=True
            Нужно ли удалить колонки "на месте" или же создать копию DataFrame.
            
        Returns
        -------
        pandas.DataFrame (optional)
            df с удалёнными колонками
    '''
    
    def bad_suffix(column):
        nonlocal suffixes
        return any(column.endswith(suffix) for suffix in suffixes)
        
    columns_to_drop = [col for col in df.columns if bad_suffix(col)]
    return df.drop(columns_to_drop, axis=1, inplace=inplace)


def extract_unique(reviews, column): 
    '''
    extract_unique(reviews, column)
        Извлекает уникальные значения из колонки в DataFrame.
        
        Parameters
        ----------
        reviews : pandas.DataFrame
            pandas.DataFrame, из которого будут извлечены значения.
        
        column : str
            Имя колонки в <reviews>.
        
        Returns
        -------
        pandas.DataFrame
            Содержит одну именованную колонку с уникальными значениями. 
    '''
    
    unique = reviews[column].unique()
    return pd.DataFrame({column: unique})


def count_unique(reviews, column):
    '''
    count_unique(reviews, column)
        Извлекает и подсчитывает уникальные значения из колонки в DataFrame.
        
        Parameters
        ----------
        reviews : pandas.DataFrame
            pandas.DataFrame, из которого будут извлечены значения.
        
        column : str
            Имя колонки в <reviews>.
        
        Returns
        -------
        pandas.DataFrame
            Содержит две колонки: с уникальными значениями и счётчиком встреченных. 
    '''
    
    return reviews[column].value_counts().reset_index(name='count').rename({'index': column}, axis=1)



def filter_reviews(reviews, users=None, orgs=None): 
    '''
    filter_reviews(reviews, users=None, orgs=None)
    Оставляет в выборке только отзывы, оставленные заданными пользователями на заданные организации. 
    
    Parameters
    ----------
        users: pandas.DataFrame, default=None
            DataFrame, содержащий колонку <user_id>.
            Если None, то фильтрация не происходит. 
            
        orgs: pandas.DataFrame, default=None
            DataFrame, содержащий колонку <org_id>.
            Если None, то фильтрация не происходит. 
    
    Returns
    -------
        pandas.DataFrame
            Отфильтрованная выборка отзывов. 

    '''
    if users is not None: 
        reviews = reviews.merge(users, on='user_id', how='inner')
        clear_df(reviews)
        
    if orgs is not None:
        reviews = reviews.merge(orgs, on='org_id', how='inner')
        clear_df(reviews)
        
    return reviews


def train_test_split(reviews, ts_start, ts_end=None):
    '''
    train_test_split(reviews, ts_start, ts_end=None)
        Разделяет выборку отзывов на две части: обучающую и тестовую. 
        В тестовую выборку попадают только отзывы с user_id и org_id, встречающимися в обучающей выборке.

        Parameters
        ----------
        reviews : pandas.DataFrame 
            Отзывы из reviews.csv с обязательными полями:
                <rating>, <ts>, <user_id>, <user_city>, <org_id>, <org_city>.

        ts_start : int
            Первый день отзывов из тестовой выборки (включительно).

        ts_end : int, default=None
            Последний день отзывов из обучающей выборки (включительно)
            Если параметр равен None, то ts_end == reviews['ts'].max(). 

        Returns
        -------
        splitting : tuple
            Кортеж из двух pandas.DataFrame такой же структуры, как и reviews:
            в первом отзывы, попавшие в обучающую выборку, во втором - в тестовую.
    '''
    
    if not ts_end:
        ts_end = reviews['ts'].max()
    
    
    reviews_train = reviews[(reviews['ts'] < ts_start) | (reviews['ts'] > ts_end)]
    reviews_test = reviews[(ts_start <= reviews['ts']) & (reviews['ts'] <= ts_end)]
    
    # 1. Выбираем только отзывы на понравившиеся места у путешественников.
    reviews_test = reviews_test[reviews_test['rating'] >= 4.0]
    user_and_org_from_different_cities = reviews_test['org_city'] != reviews_test['user_city']
    reviews_test = reviews_test[user_and_org_from_different_cities]
    
    # 2. Оставляем в тесте только тех пользователей и организации, которые
    #   встречались в трейне.
    train_orgs = extract_unique(reviews_train, 'org_id')
    train_users = extract_unique(reviews_train, 'user_id')
    
    reviews_test = filter_reviews(reviews_test, orgs=train_orgs)

    return reviews_train, reviews_test


def process_reviews(reviews):
    '''
    process_reviews(reviews)
        Извлекает из набора отзывов тестовых пользователей и таргет. 
        
        Parameters
        ----------
        reviews : pandas.DataFrame
            DataFrame с отзывами, содержащий колонки <user_id> и <org_id>
        
        Returns
        -------
        X : pandas.DataFrame
            DataFrame такой же структуры, как и в test_users.csv
            
        y : pandas.DataFrame
            DataFrame с колонками <user_id> и <target>. 
            В <target> содержится список org_id, посещённых пользователем. 
    '''
    
    y = reviews.groupby('user_id')['org_id'].apply(list).reset_index(name='target')
    X = pd.DataFrame(y['user_id'])
    
    return X, y

# %%
train_reviews, test_reviews = train_test_split(reviews, 1116)
X_test, y_test = process_reviews(test_reviews)

# %% [markdown]
# Посмотрим, сколько всего уникальных пользователей попало в эту тестовую выборку:

# %%
len(X_test)


# %% [markdown]
# # Выбор метрики
# Для последующей оценки качества следует выбрать метрику.

# %% [markdown]
# Метрика **Precision**:
#
# Доля верно предсказанных классификатором положительных объектов, из всех
# объектов, которые классификатор верно или неверно определил как
# положительные.
# Принимает на вход два DataFrame, имеющих такую же структуру, как и **y_test**.
# В контексте рекомендательных систем формула выглядит
# следующим образом: 
# $$
# \begin{align*}
#   \qquad P = \frac{\textrm{Количество релевантных рекомендаций}}{\textrm{Количество рекомендаций}}
# \end{align*}
# $$
# К сожалению, данная метрика не учитывает порядок рекомендаций. Нам же хочется
# предлагать пользователю в первую очередь те места, в которых мы наиболее
# уверены. Для этого лучше предназначены следующие метрики [2]:
#
# **Average Precision**:
#
# $$
# \begin{align*}
#   \textrm{AP@N} = \frac{1}{m}\sum_{k=1}^N \textrm{($P(k)$ если $k_{ая}$ рекомендация была релевантна)} = \frac{1}{m}\sum_{k=1}^N P(k)\cdot rel(k),
# \end{align*}
# $$
# где m - количество рекомендаций, N - количество рекомендаций, которые мы
# берём в расчёт $ N \leq m $.
#
# В конечном итоге, нас интересует **Mean Average Precision**  - среднее Average
# Precision по пользователям:
# $$
# \begin{align*}
#   \textrm{MAP@N} = \frac{1}{|U|}\sum_{u=1}^|U|(\textrm{AP@N})_u = \frac{1}{|U|} \sum_{u=1}^|U| \frac{1}{m}\sum_{k=1}^N P_u(k)\cdot rel_u(k).
# \end{align*}
# $$


# %%
def mean_average_precision_at_n(size=20):
    '''
    mean_average_precision_at_n(size=20)
        Создаёт метрику под <size> сделанных предсказаний.
        
        Parameters
        ----------
        size : int, default=20
            Размер рекомендованной выборки для каждого пользователя
        
        Returns
        -------
        func(pd.DataFrame, pd.DataFrame) -> float
            Функция, вычисляющая mean_average_precision_at_n.
        
    '''
    
    assert size >= 1, "Size must be greater than zero!"
    
    def metric(y_true, predictions, size=size):
        '''
        metric(y_true, predictions, size=size)
            Метрика mean_average_precision_at_n для двух перемешанных наборов <y_true> и <y_pred>.
            
            Parameters
            ----------
            y_true : pd.DataFrame
                DataFrame с колонками <user_id> и <target>. 
                В <target> содержится список настоящих org_id, посещённых пользователем. 
                
            predictions : pd.DataFrame
                DataFrame с колонками <user_id> и <target>. 
                В <target> содержится список рекомендованных для пользователя org_id.
                
            Returns
            -------
            float 
                Значение метрики.
        '''
        
        y_true = y_true.rename({'target': 'y_true'}, axis='columns')
        predictions = predictions.rename({'target': 'predictions'}, axis='columns')
        
        merged = y_true.merge(predictions, left_on='user_id', right_on='user_id')
    
        def average_precision_for_user(x: pd.DataFrame, size=size) -> float:
            '''
            average_precision_for_user(x: pd.DataFrame, size=size) -> float
            Средняя точность для пользователя.

            Parameters
            ----------
            x : pd.DataFrame
                DataFrame с колонками <user_id>, <y_true>, <predictions>.

            Returns
            -------
            float
                Значение от 0 до 1 - средняя точность для пользователя.
            '''
            y_true = x[1][1]
            predictions = x[1][2]
            
            weight = 0
            
            inner_weights = [0]
            for n, item in enumerate(predictions):
                inner_weight = inner_weights[-1] + (1 if item in y_true else 0)
                inner_weights.append(inner_weight)
            
            for n, item in enumerate(predictions):                
                if item in y_true:
                    weight += inner_weights[n + 1] / (n + 1)
                    
            return weight / min(len(y_true), size)

        return np.mean([average_precision_for_user(user_row) for user_row in merged.iterrows()])
    
        
    return metric


def print_score(score):
    print(f"Score: {score:.6f}")
    
    
N = 20
mapN = mean_average_precision_at_n(N)

# %% [markdown]
# # Подбор рекомендаций
# Выделяют несколько типов таких систем, которые можно разделить по подходу к рекомендациям [1]:
# 1. Content-based
# Пользователю рекомендуются объекты, похожие на те, которые этот пользователь уже использовал или просматривал ранее.
# Степень схожести объектов оценивается по их характеристикам.
#
# 2. Коллаборативная фильтрация (Collaborative Filtering)
# Пользователю рекомендуются объекты, подобранные в соответствии с оценками, поставленными ранее.
# Для рекомендации используется история оценок как самого пользователя, так и других пользователей.
# Данный подход основан на оценках, которые поставили пользователи. Он считается наиболее простым в реализации и чаще всего используется в недорогих рекомендательных системах.
# Для рекомендации пользователю достопримечательности предсказываются оценки,
# которые он может поставить, и выбираются места с наилучшими оценками.
# Существует два способа реализации коллаборативной фильтрации [1]:
# - User-based: здесь определяются пользователи, наиболее похожие по оценкам на того, которому составляется рекомендация, и в соответствии с их оценками предсказываются оценки текущего пользователя.
# - Item-based: этот способ похож на User-based. Разница же заключается в том, что здесь
# определяются места, наиболее похожие по оценкам на место, оценку которого мы
# хотим предсказать. В соответствии с оценками этих мест предсказывается оценка
# на данную достопримечательность.
# Разрабатываемые системы, основанные User-based и Item-based коллаборативной
# фильтрации, будут рекомендовать пользователям места, которым пользователи поставили бы наивысшую оценку.
#
# 3. Гибридные системы
# Этот подход представляет собой совокупность двух предыдущих.

# %% [markdown]
# Результаты будем записывать в словарь:

# %%
results = {}

# %% [markdown]
# # Подходы без машинного обучения

# %% [markdown]
# ## Случайные N мест

# %% [markdown]
# Попробуем предлагать пользователям случайные места из другого города. 

# %%
spb_orgs = orgs[orgs['city'] == 'spb']['org_id']
msk_orgs = orgs[orgs['city'] == 'msk']['org_id']

test_users_with_locations = X_test.merge(users, on='user_id')

# %%
# %%time

np.random.seed(1337)
choose = lambda x: np.random.choice(spb_orgs, N) if x['user_id'] == 'msk' else np.random.choice(msk_orgs, N)
target = test_users_with_locations.apply(choose, axis=1)

predictions = X_test.copy()
predictions['target'] = target

mapN_random_result = mapN(y_test, predictions)
results['random'] = mapN_random_result
print_score(mapN_random_result)

# %% [markdown]
# ## N самых популярных мест

# %% [markdown]
# Предыдущий подход, очевидно, не очень удачно предсказывает, какие места посетит пользователей. 
#
# Попробуем улучшить стратегию: будем предлагать пользователям самые популярные места, то есть те, на которые оставлено больше всего отзывов. 

# %%
msk_orgs = train_reviews[(train_reviews['rating'] >= 4) & (train_reviews['org_city'] == 'msk')]['org_id']
msk_orgs = msk_orgs.value_counts().index[:N].to_list()

spb_orgs = train_reviews[(train_reviews['rating'] >= 4) & (train_reviews['org_city'] == 'spb')]['org_id']
spb_orgs = spb_orgs.value_counts().index[:N].to_list()

# %%
# %%time

choose = lambda x: spb_orgs if x['user_id'] == 'msk' else msk_orgs
target = test_users_with_locations.apply(choose, axis=1)

predictions = X_test.copy()
predictions['target'] = target

mapN_popularity_result = mapN(y_test, predictions)
results['popularity'] = mapN_popularity_result
print_score(mapN_popularity_result)

# %% [markdown]
# Отлично, метрика немного улучшилась. Но стоит попробовать доработать эту тактику. 

# %% [markdown]
# ## N самых популярных мест среди туристов

# %%
tourist_reviews = train_reviews[train_reviews['rating'] >= 4.0]

# набор отзывов только от туристов
tourist_reviews = tourist_reviews[tourist_reviews['user_city'] != tourist_reviews['org_city']]

# выбираем самые популярные места среди туристов из Москвы и Питера
msk_orgs = tourist_reviews[tourist_reviews['org_city'] == 'msk']['org_id']
msk_orgs = msk_orgs.value_counts().index[:N].to_list()

spb_orgs = tourist_reviews[tourist_reviews['org_city'] == 'spb']['org_id']
spb_orgs = spb_orgs.value_counts().index[:N].to_list()

# %%
# %%time

choose = lambda x: spb_orgs if x['user_id'] == 'msk' else msk_orgs
target = test_users_with_locations.apply(choose, axis=1)

predictions = X_test.copy()
predictions['target'] = target

mapN_tourist_popularity_result = mapN(y_test, predictions)
results['tourist_popularity'] = mapN_tourist_popularity_result
print_score(mapN_tourist_popularity_result)


# %% [markdown]
# Метрика улучшилась ещё немного.

# %% [markdown]
# ## N / rubrics_count самых популярных мест из каждой рубрики

# %%
def extract_top_by_rubrics(reviews, N):
    '''
    extract_top_by_rubrics(reviews, N)
        Набирает самые популярные организации по рубрикам, сохраняя распределение.
        
        Parameters
        ----------
        reviews : pd.DataFrame
            Отзывы пользователей для рекомендации.
            
        N : int
            Число рекомендаций.
        
        Returns
        -------
        orgs_list : list
            Список отобранных организаций.
    '''
    
    # Извлечение популярных рубрик.
    reviews = reviews.merge(orgs, on='org_id')[['org_id', 'rubrics_id']]
    
    rubrics = reviews.explode('rubrics_id').groupby('rubrics_id').size()
    rubrics = (rubrics / rubrics.sum() * N).apply(round).sort_values(ascending=False)

    # Вывод списка рубрик по убыванию популярности.
    print(
        pd.read_csv('data/rubrics.csv', sep=',')
        .merge(rubrics.reset_index(), left_index=True, right_on='rubrics_id')
        .sort_values(by=0, ascending=False)[['rubric_id', 0]]
    )
    
    # Извлечение популярных организаций.
    train_orgs = reviews.groupby('org_id').size().reset_index(name='count').merge(orgs, on='org_id')
    train_orgs = train_orgs[['org_id', 'count', 'rubrics_id']]

    most_popular_rubric = lambda rubrics_id: max(rubrics_id, key=lambda rubric_id: rubrics[rubric_id])
    train_orgs['rubrics_id'] = train_orgs['rubrics_id'].apply(most_popular_rubric)
    
    orgs_by_rubrics = train_orgs.sort_values(by='count', ascending=False).groupby('rubrics_id')['org_id'].apply(list)
    
    # Соберём самые популярные организации в рубриках в один список.
    
    orgs_list = []

    for rubric_id, count in zip(rubrics.index, rubrics):
        if rubric_id not in orgs_by_rubrics:
            continue 

        orgs_list.extend(orgs_by_rubrics[rubric_id][:count])
    
    return orgs_list


msk_orgs = extract_top_by_rubrics(tourist_reviews[tourist_reviews['org_city'] == 'msk'], N)
spb_orgs = extract_top_by_rubrics(tourist_reviews[tourist_reviews['org_city'] == 'spb'], N)

# %%
# %%time

choose = lambda x: spb_orgs if x['user_id'] == 'msk' else msk_orgs
target = test_users_with_locations.apply(choose, axis=1)

predictions = X_test.copy()
predictions['target'] = target

mapN_popular_rubrics_result = mapN(y_test, predictions)
results['popular_rubrics'] = mapN_popular_rubrics_result
print_score(mapN_popular_rubrics_result)


# %% [markdown]
# # ML методы.

# %% [markdown]
# # Memory-based. Alternating Least Squares.
# Для этой группы методов требуется явное построение матрицы __пользователь-организация__ (__interaction matrix__), где на пересечении $i$-ой строки и $j$-ого столбца будет рейтинг, который $i$-ый пользователь выставил $j$-ой организации или же пропуск, если рейтинг не был установлен. 

# Метод ALS является более продвинутым по сравнению с другими методами.
# Он также использует факторизацию матриц, однако существует возможность
# параллельного исполнения алгоритма. По этой причине этот метод зачастую
# используют в индустрии для генерации рекомендаций, например, Netflix за
# улучшение своего алгоритма коллаборативной фильтрации давали 1.000.000 $ -
# именно алгоритм, основанный на ALS помог авторам достичь необходимых
# результатов в предсказании со значением метрики в 0.1006 (RMSE) [3].


# %% [markdown]
# # Сравнение результатов

# %%
results_ax = pd.DataFrame([results]).plot(y=['random', 'popularity', 'tourist_popularity', 'popular_rubrics'], kind='bar', figsize=(8,8))
results_ax.set_xlabel('Методы')
results_ax.set_ylabel('Значение метрики MAP@N')


# %% [markdown]
# Источники
# 1. K. Falk, Practical recommender systems. Shelter Island, NY: Manning, 2019.
# 2. «Mean Average Precision (MAP) For Recommender Systems», Evening Session, 13:30:00-04:00 г. https://sdsawtelle.github.io/blog/output/./mean-average-precision-MAP-for-recommender-systems.html (просмотрено 5 июнь 2022 г.).
# 3. «Netflix Prize», Wikipedia. 16 апрель 2022 г. Просмотрено: 6 июнь 2022 г. [Онлайн]. Доступно на: https://en.wikipedia.org/w/index.php?title=Netflix_Prize&oldid=1083006521

