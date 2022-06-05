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


# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import sparse
from sklearn.metrics import average_precision_score
from tqdm.notebook import tqdm

from implicit.als import AlternatingLeastSquares

from typing import List

# %matplotlib inline

# %% [markdown]
# # Сырые данные

# %% [markdown]
# ## Считываем данные из .csv

# %% [markdown]
# Некоторые данные (такие как рубрики и признаки), представлены строками значений. Преобразуем их в списки чисел. 

# %%
# Переводит строку вида '1 2 3 4' в список [1, 2, 3, 4].
to_int_list = lambda values: [int(value) for value in str(values).split(' ')]

def apply_to_columns(df: pd.DataFrame, columns: List[str], func=to_int_list):
    """Apply function to the specified columns of the dataframe.
    """

    for column in columns:
        df.loc[~df[column].isnull(), column] = df.loc[
            ~df[column].isnull(), column
        ].apply(func)


# %% [markdown]
# В первую очередь нам понадобятся данные по **пользователям**, **организациям** и сами **отзывы**. 

# %%
users = pd.read_csv('data/users.csv')
users.head()

# %%
orgs = pd.read_csv('data/organisations.csv')

# %%
# create lists
columns = ['rubrics_id', 'features_id']
apply_to_columns(orgs, columns)

orgs.head()

# %% [markdown]
# Чтобы не делать __join__ каждый раз, когда нам потребуется узнать, из какого города организация или пользователь, сразу добавим эту информацию в отзывы.

# %%
# Читаем датасет по частям, затем соединяем в один. Позволяет загружать большие
#   (500 мб и более) датасеты.
#   К сожалению, low_memory=True дропает кернел, несмотря на то что внутри
# использует схожий механизм.
# - Количество рядов, содеражихся в одной части датасета.
chunksize = 1000
reviews = pd.concat(pd.read_csv('data/reviews.csv', chunksize=chunksize))
reviews

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
#
# Посмотрим на распределение новых отзывов по дням, чтобы понять, как лучше организовать валидацию. 

# %%
sns.displot(data=reviews, x='ts', height=8)
plt.title('Распределение отзывов по дням')
plt.show()


# %% [markdown]
# # Train-test split

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
    
    # 1. Выбираем только отзывы на понравившиеся места у путешественников
    reviews_test = reviews_test[reviews_test['rating'] >= 4.0]
    user_and_org_from_different_cities = reviews_test['org_city'] != reviews_test['user_city']
    reviews_test = reviews_test[user_and_org_from_different_cities]
    
    # 2. Оставляем в тесте только тех пользователей и организации, которые встречались в трейне
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
reviews['ts'].max()

# %% [markdown]
# Всего в выборку попали отызывы за **1216** дней. 
#
# Отложим в тестовую выборку отзывы за последние **100** дней.

# %%
train_reviews, test_reviews = train_test_split(reviews, 1116)
X_test, y_test = process_reviews(test_reviews)

# %% [markdown]
# Посмотрим, сколько всего уникальных пользователей попало в эту тестовую выборку:

# %%
len(X_test)


# %% [markdown]
# # Метрика

# %% [markdown]
# Метрика принимает на вход два DataFrame, имеющих такую же структуру, как и **y_test**.
# Вычисляется Precision, в контексте рекомендательных систем формула выглядит
# следующим образом: 
# $$
# \begin{align*}
#   \qquad P = \frac{\textrm{Количество релевантных рекомендаций}}{\textrm{Количество рекомендаций}}
# \end{align*}
# $$
#
# Далее берётся Average Precision:
#
# $$
# \begin{align*}
#   \textrm{AP@N} = \frac{1}{m}\sum_{k=1}^N \textrm{($P(k)$ если $k_{ая}$ рекомендация была релевантна)} = \frac{1}{m}\sum_{k=1}^N P(k)\cdot rel(k),
# \end{align*}
# $$
# где m - количество рекомендаций, N - количество рекомендаций, которые мы
# берём в расчёт $ N \leq m $
#
# В конечном итоге, нас интересует Mean Average Precision - среднее Average
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

print_score(mapN(y_test, predictions))

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

print_score(mapN(y_test, predictions))

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

print_score(mapN(y_test, predictions))


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
        pd.read_csv('data/rubrics.csv')
        .merge(rubrics.reset_index(), left_index=True, right_on='rubrics_id')
        .sort_values(by=0, ascending=False)[['rubric_id', 0]]
    )
    
    # извлечение популярных организаций
    train_orgs = reviews.groupby('org_id').size().reset_index(name='count').merge(orgs, on='org_id')
    train_orgs = train_orgs[['org_id', 'count', 'rubrics_id']]

    most_popular_rubric = lambda rubrics_id: max(rubrics_id, key=lambda rubric_id: rubrics[rubric_id])
    train_orgs['rubrics_id'] = train_orgs['rubrics_id'].apply(most_popular_rubric)
    
    orgs_by_rubrics = train_orgs.sort_values(by='count', ascending=False).groupby('rubrics_id')['org_id'].apply(list)
    
    # соберём самые популярные организации в рубриках в один список
    
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

print_score(mapN(y_test, predictions))


# %% [markdown]
# # ML методы. Коллаборативная фильтрация

# %% [markdown]
# ## Memory-based
#
# Для этой группы методов требуется явное построение матрицы __пользователь-организация__ (__interaction matrix__), где на пересечении $i$-ой строки и $j$-ого столбца будет рейтинг, который $i$-ый пользователь выставил $j$-ой организации или же пропуск, если рейтинг не был установлен. 

# %%
def reduce_reviews(reviews, min_user_reviews=5, min_org_reviews=13):
    '''
    reduce_reviews(reviews, min_user_reviews=5, min_org_reviews=13)
        Убирает из выборки пользователей и организации, у которых менее <min_reviews> отзывов в родном городе. 
        Оставляет только отзывы туристов. 
        
        Parameters
        ----------
        reviews : pandas.DataFrame 
            Выборка отзывов с обязательными полями:
                <user_id>, <user_city>.
        
        min_user_reviews : int, default=5
            Минимальное количество отзывов у пользователя, необходимое для включения в выборку.
            
        min_org_reviews : int, default=13
            Минимальное количество отзывов у организации, необходимое для включения в выборку.
            
        Returns
        -------
        splitting : tuple
            Кортеж из двух наборов.
            Каждый набор содержит 2 pandas.DataFrame:
                1. Урезанная выборка отзывов
                2. Набор уникальных организаций
                
            Первый набор содержит DataFrame-ы, относящиеся к отзывам, оставленным в родном городе, а второй -
            к отзывам, оставленным в чужом городе. ё
            
        users : pd.DataFrame
            Набор уникальных пользователей в выборке
        
    '''
    
    inner_reviews = reviews[reviews['user_city'] == reviews['org_city']]
    outer_reviews = reviews[reviews['user_city'] != reviews['org_city']]

    # оставляем только отзывы туристов на родной город 
    tourist_users = extract_unique(outer_reviews, 'user_id')
    inner_reviews = filter_reviews(inner_reviews, users=tourist_users)
    
    # выбираем только тех пользователей и организации, у которых есть <min_reviews> отзывов
    top_users = count_unique(inner_reviews, 'user_id')
    top_users = top_users[top_users['count'] >= min_user_reviews]
        
    top_orgs = count_unique(inner_reviews, 'org_id')
    top_orgs = top_orgs[top_orgs['count'] >= min_org_reviews]
        
    inner_reviews = filter_reviews(inner_reviews, users=top_users, orgs=top_orgs)
    outer_reviews = filter_reviews(outer_reviews, users=top_users)
    
    # combine reviews
    reviews = pd.concat([inner_reviews, outer_reviews])
    users = extract_unique(reviews, 'user_id')
    orgs = extract_unique(reviews, 'org_id')
    
    
    return (
        (
            inner_reviews,
            extract_unique(inner_reviews, 'org_id')
        ),
        (
            outer_reviews,
            extract_unique(outer_reviews, 'org_id')
        ),
        extract_unique(inner_reviews, 'user_id')
    )


# %%
def create_mappings(df, column):
    '''
    create_mappings(df, column)
        Создаёт маппинг между оригинальными ключами словаря и новыми порядковыми.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame с данными.
            
        column : str
            Название колонки, содержащей нужны ключи. 
        
        Returns
        -------
        code_to_idx : dict
            Словарь с маппингом: "оригинальный ключ" -> "новый ключ".
        
        idx_to_code : dict
            Словарь с маппингом: "новый ключ" -> "оригинальный ключ".
    '''
    
    code_to_idx = {}
    idx_to_code = {}
    
    for idx, code in enumerate(df[column].to_list()):
        code_to_idx[code] = idx
        idx_to_code[idx] = code
        
    return code_to_idx, idx_to_code


def map_ids(row, mapping):
    '''
    Вспомогательная функция
    '''
    
    return mapping[row]


def interaction_matrix(reviews, test_users, min_user_reviews=5, min_org_reviews=12): 
    '''
    interaction_matrix(reviews, test_users, min_user_reviews=5, min_org_reviews=12)
        Создаёт блочную матрицу взаимодействий (вид матрицы описан в Returns)
        
        Parameters
        ----------
        reviews : pd.DataFrame
            Отзывы пользователей для матрицы взаимодействий.
            
        test_users : pd.DataFrame
            Пользователи, для которых будет выполнятся предсказание. 
        
        min_user_reviews : int, default=5
            Минимальное число отзывов от пользователя, необходимое для включения его в матрицу.
        
        min_org_reviews : int, default=12
            Минимальное число отзывов на организацию, необходимое для включения её в матрицу.
    
        Returns
        -------
        InteractionMatrix : scipy.sparse.csr_matrix
            Матрица, содержащая рейтинги, выставленные пользователями.
            Она блочная и имеет такой вид:
                 ---------------------------------------------------
                | TRAIN USERS, INNER ORGS | TRAIN USERS, OUTER ORGS |
                |                         |                         |
                 ---------------------------------------------------
                |  TEST USERS, INNER ORGS |  TEST USERS, OUTER ORGS |
                |                         |                         |
                 ---------------------------------------------------

        splitting : tuple
            Кортеж, содержащий два целых числа: 
                1. Число пользователей в обучающей выборке 
                2. Число организаций в домашнем регионе

        splitting: tuple
            Кортеж, содержащий два котрежа из двух словарей:
                1. (idx_to_uid, uid_to_idx) - содержит маппинг индекса к user_id
                2. (idx_to_oid, oid_to_idx) - содержит маппинг индекса к org_id
    '''
    
    
    info = reduce_reviews(train_reviews, min_user_reviews, min_org_reviews)
    (inner_reviews, inner_orgs), (outer_reviews, outer_orgs), train_users = info
    
    # удалим из обучающей выборки пользователей, которые есть в тестовой
    test_users = test_users[['user_id']]
    
    train_users = (
        pd.merge(train_users, test_users, indicator=True, how='outer')
        .query('_merge=="left_only"')
        .drop('_merge', axis=1)
    )
    
    inner_reviews = filter_reviews(inner_reviews, train_users)
    outer_reviews = filter_reviews(outer_reviews, train_users)
    
    # оставляем отзывы, оставленные тестовыми пользователями
    test_reviews = filter_reviews(reviews, test_users, pd.concat([inner_orgs, outer_orgs]))
    
    # получаем полный набор маппингов
    all_users = pd.concat([train_users, test_users])
    all_orgs = pd.concat([inner_orgs, outer_orgs])
    
    uid_to_idx, idx_to_uid = create_mappings(all_users, 'user_id')
    oid_to_idx, idx_to_oid = create_mappings(all_orgs, 'org_id')
    
    # собираем матрицу взаимодействий 
    reviews = pd.concat([inner_reviews, outer_reviews, test_reviews])    
        
    I = reviews['user_id'].apply(map_ids, args=[uid_to_idx]).values
    J = reviews['org_id'].apply(map_ids, args=[oid_to_idx]).values
    values = reviews['rating']
        
    interactions = sparse.coo_matrix(
        (values, (I, J)), 
        shape=(len(all_users), len(all_orgs)), 
        dtype=np.float64
    ).tocsr()
    
    
    return (
        interactions, 
        (len(train_users), len(inner_orgs)), 
        (
            (idx_to_uid, uid_to_idx),
            (idx_to_oid, oid_to_idx)
        )
    )

# %% [markdown]
# ## Alternating Least Squares
# Метод не столь точный как градиентный спуск, но достаточно эффективный:
# главным достоинством является скорость, даже на больших данных показывает
# хорошие результаты [1].

# %%
# %%time
def make_predictions(interactions, X_test, N):
    '''
    make_predictions(interactions, X_test, N)
        Делает рекомендации для пользователей из <X_test> на основе матрицы взаимодействий. 
        
        Parameters
        ----------
        interactions : scipy.sparse.csr_matrix
            Разреженная матрица взаимодействий.
            
        X_test : pd.DataFrame
            Набор тестовых пользователей, для которых нужно сделать рекомендации. 
        
        N : int
            Число рекомендаций для каждого пользователя. 
        
        Returns
        -------
        predictions : pd.DataFrame
            DataFrame с колонками <user_id> и <target>. 
            В <target> содержится список рекомендованных для пользователя org_id.
        
        
    '''
    
    predictions = X_test[['user_id']].copy()
    predictions['target'] = pd.Series(dtype=object)
    predictions = predictions.set_index('user_id')
    
    interactions, (train_users_len, inner_orgs_len), mappings = interactions
    (idx_to_uid, uid_to_idx), (idx_to_oid, oid_to_idx) = mappings

    base_model = AlternatingLeastSquares(
        factors=5, 
        iterations=75, 
        regularization=0.05, 
        random_state=42
    )
    
    base_model.fit(interactions.T)
    
    orgs_to_filter = list(np.arange(inner_orgs_len))

    recommendations = base_model.recommend_all(
        interactions,
        N=N,
        filter_already_liked_items=True,
        filter_items=orgs_to_filter,
        show_progress=True
    )
    
    for user_id in tqdm(X_test['user_id'].values, leave=False):
        predictions.loc[user_id, 'target'] = list(
            map(
                lambda org_idx: idx_to_oid[org_idx], 
                recommendations[uid_to_idx[user_id]]
            )
        )
        
    return predictions.reset_index()


msk_interactions = interaction_matrix(
    train_reviews[train_reviews['user_city'] == 'msk'],
    test_users_with_locations[test_users_with_locations['city'] == 'msk'],
)

spb_interactions = interaction_matrix(
    train_reviews[train_reviews['user_city'] == 'spb'],
    test_users_with_locations[test_users_with_locations['city'] == 'spb'],
)       
        
test_msk_users = test_users_with_locations[test_users_with_locations['city'] == 'msk']
test_spb_users = test_users_with_locations[test_users_with_locations['city'] == 'spb']

msk_predictions = make_predictions(msk_interactions, test_msk_users, N)
spb_predictions = make_predictions(spb_interactions, test_spb_users, N)

predictions = pd.concat([msk_predictions, spb_predictions])

# %%
# %%time

print_score(mapN(y_test, predictions))

# %% [markdown]
# Источники
# 1. **Ким Фалк** - Практические рекомендательные системы.

# %% [markdown]
# # Submission
#
# Выберем лучший метод на валидации, переобучим его на всей выборке и сделаем предсказание на тестовой выборке. 

# %% [markdown]
# ## Without ML

# %%
test_users = pd.read_csv('data/test_users.csv')
test_users['city'] = test_users.merge(users, on='user_id')['city']

tourist_reviews = reviews[(reviews['ts'] > 900) & (reviews['user_city'] != reviews['org_city'])]

msk_orgs = extract_top_by_rubrics(tourist_reviews[tourist_reviews['org_city'] == 'msk'], N)
spb_orgs = extract_top_by_rubrics(tourist_reviews[tourist_reviews['org_city'] == 'spb'], N)

msk_orgs = str(' '.join(map(str, msk_orgs)))
spb_orgs = str(' '.join(map(str, spb_orgs)))

# %%
choose = lambda x: spb_orgs if x['user_id'] == 'msk' else msk_orgs
target = test_users.apply(choose, axis=1)

predictions = test_users[['user_id']]
predictions['target'] = target

predictions.head()

# %%
predictions.to_csv('answers.csv', index=None)

# %% [markdown]
# ## With ML

# %%
test_users = pd.read_csv('data/test_users.csv')
test_users = test_users.merge(users, on='user_id')


test_msk_users = test_users[test_users['city'] == 'msk'][['user_id', 'city']]
test_spb_users = test_users[test_users['city'] == 'spb'][['user_id', 'city']]


msk_interactions = interaction_matrix(
    reviews[reviews['user_city'] == 'msk'],
    test_msk_users
)

spb_interactions = interaction_matrix(
    reviews[reviews['user_city'] == 'spb'],
    test_spb_users
)

msk_predictions = make_predictions(msk_interactions, test_msk_users, N)
spb_predictions = make_predictions(spb_interactions, test_spb_users, N)

predictions = pd.concat([msk_predictions, spb_predictions])

# %%
predictions['target'] = predictions['target'].apply(lambda orgs: ' '.join(map(str, orgs)))
predictions.head()

# %%
predictions.to_csv('answers_ml.csv', index=None)

# %%

# %%

# %%

# %%

# %%
import requests 

# %%
r = requests.get('https://api.github.com/events')

# %%
import json 
events = json.loads(r.text)

