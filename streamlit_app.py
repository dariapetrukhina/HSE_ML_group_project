import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Разделы меню
sections = {
    "Загрузка данных": "load_data",
    "Разведочный анализ": "analize",
    "Работа с признаками": "drop_columns",
    "Выбор и обучение моделей": "learning",
    "Оценка качетсва модели": "Quality_control"
}
# Отображение бокового меню с ссылками на разделы кода
st.sidebar.header("Боковое меню")
for section_name, section_id in sections.items():
    st.sidebar.markdown(f"- [{section_name}](#{section_id})")


# Заголовок
st.title("Music genre prediction")
st.success("Участники команды: Иван, Дарья, Юлия, Михаил")

st.write("<h2 id='load_data'>Загрузка данных</h2>", unsafe_allow_html=True)


# Загрузка данных
st.write("Загрузка данных")
st.write("Train = https://www.dropbox.com/scl/fi/5zy935lqpaqr9lat76ung/music_genre_train.csv?rlkey=ccovu9ml8pfi9whk1ba26zdda&dl=1")
st.write("Test = https://www.dropbox.com/scl/fi/o6mvsowpp9r3k2lejuegt/music_genre_test.csv?rlkey=ac14ydue0rzlh880jwj3ebum4&dl=1")
TRAIN = "https://www.dropbox.com/scl/fi/5zy935lqpaqr9lat76ung/music_genre_train.csv?rlkey=ccovu9ml8pfi9whk1ba26zdda&dl=1"
TEST = "https://www.dropbox.com/scl/fi/o6mvsowpp9r3k2lejuegt/music_genre_test.csv?rlkey=ac14ydue0rzlh880jwj3ebum4&dl=1"
train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

RANDOM_STATE = 42
TEST_SIZE = 0.25

# Вывод случайных строк из обучающего и тестового наборов данных
st.write("Примеры данных из обучающего набора:")
st.write(train.sample(5))

st.write("Примеры данных из тестового набора:")
st.write(test.sample(5))

# Описание полей данных
field_descriptions = {
    "instance_id": "Уникальный идентификатор трека",
    "track_name": "Название трека",
    "acousticness": "Акустичность",
    "danceability": "Танцевальность",
    "duration_ms": "Продолжительность в миллисекундах",
    "energy": "Энергичность",
    "instrumentalness": "Инструментальность",
    "key": "Тональность",
    "liveness": "Привлекательность",
    "loudness": "Громкость",
    "mode": "Наклонение",
    "speechiness": "Выразительность",
    "tempo": "Темп",
    "obtained_date": "Дата загрузки в сервис",
    "valence": "Привлекательность произведения для пользователей сервиса",
    "music_genre": "Музыкальный жанр"
}
# Отображение описания полей данных
st.write("Описание полей данных:")
for field, description in field_descriptions.items():
    st.write(f"- **{field}:** {description}")

# Создание DataFrame
df = pd.concat([test, train])
df_orig = df.copy()

# Отображение типов данных
st.write("Типы данных:")
st.write(df.dtypes)

# Подсчет количества значений каждого типа данных
st.write("Количество столбцов каждого типа данных:")
st.write(df.dtypes.value_counts())

# Отображение формы DataFrame
st.write("Форма DataFrame:")
st.write(df.shape)

# Подсчет количества нулевых значений в столбце "music_genre"
st.write("Количество нулевых значений в столбце 'music_genre':")
st.write(df.music_genre.isnull().sum())

# Отображение распределения значений в столбце "music_genre"
st.write("Распределение значений в столбце 'music_genre':")
st.write(df.music_genre.value_counts())


# Распределение значений в столбце "music_genre" с нормализацией
st.write("Распределение значений в столбце 'music_genre' с нормализацией:")
st.write(df.music_genre.value_counts(normalize=True))

# Количество уникальных значений в столбце "instance_id"
st.write("Количество уникальных значений в столбце 'instance_id':")
st.write(df.instance_id.nunique())

# Количество уникальных значений в каждом столбце
st.write("Количество уникальных значений в каждом столбце:")
for col in df.columns:
    st.write(f"В поле {col} кол-во уникальных значений: {df[col].nunique()}")

# Анализ числовых переменных
describe = df.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.95]).T.reset_index()
describe['share NA'] = 1 - describe['count'] / len(df)

# Отображение результатов анализа числовых переменных
st.write("Анализ числовых переменных:")
st.write(describe)



st.write("<h2 id='load_data'>Загрузка данных</h2>")



# Анализ категориальных переменных
describe_cat = df.describe(include=['object']).T.reset_index()
describe_cat['share NA'] = 1 - describe_cat['count'] / len(df)
describe_cat['freq share'] = describe_cat['freq'] / describe_cat['count']

# Отображение результатов анализа категориальных переменных
st.write("Анализ категориальных переменных:")
st.write(describe_cat)

# Анализ уникальных значений категориальных факторов
categorical = list(df.dtypes[df.dtypes == "object"].index)
unique_values = {}
for col in categorical:
    unique_values[col] = df[col].unique()

# Отображение результатов анализа уникальных значений категориальных факторов
st.write("Уникальные значения категориальных факторов:")
for col, values in unique_values.items():
    st.write(f'В поле {col} следующие уникальные значения: {values}')

# Получение списка категориальных переменных, исключая "music_genre"
categorical = list(df.dtypes[df.dtypes == "object"].index)
categorical.remove('music_genre')

# Отображение списка категориальных переменных
st.write("Список категориальных переменных (исключая 'music_genre'):")
st.write(categorical)


# Заполнение пропусков у категориальных факторов наиболее часто встречающимся значением
for column in categorical:
    most_common_value = df[column].mode()[0]
    df[column].fillna(most_common_value, inplace=True)

# Отображение информации о заполнении пропусков
st.write("Пропуски в категориальных факторах заполнены наиболее часто встречающимся значением.")


# Удаление ненужных полей
df = df.drop(columns=['instance_id', 'obtained_date'])

# Отображение размера DataFrame после удаления
st.write(f"Размер DataFrame после удаления ненужных полей: {df.shape}")

# Выделение числовых полей
numerical = list(df.dtypes[df.dtypes == "float64"].index)

# Отображение числовых полей
st.write("Числовые поля:", numerical)

# Заполнение пропусков у числовых факторов средним
for column in numerical:
    mean_value = df[column].mean()
    df[column] = df[column].fillna(mean_value)

# Отображение информации о пропущенных значениях после заполнения
st.write("Количество пропущенных значений после заполнения:", df.isnull().sum())


st.write("Разведочный анализ")

# Гистограмма факторов на всем датафрейме
fig, axes = plt.subplots(5, 2, figsize=(10, 10))

# Перебираем каждый столбец и строим гистограмму на соответствующем подграфике
for i, col in enumerate(numerical):
    x = i % 5
    y = i // 5
    df[col].plot.hist(ax=axes[x, y])
    axes[x, y].set_title(col)

plt.tight_layout()

# Отображаем графика
st.pyplot(fig)


st.write(describe)


# Найти строки, где значение 'duration_ms' равно 4497994
df[df['duration_ms'] == 4497994]

# Определить количество строк, где 'duration_ms' равно -1 и 'music_genre' не является NaN
len(df[(df['duration_ms'] == -1) & (df['music_genre'].notna())])

# Заменить значение -1 в поле 'duration_ms' средним значением по жанру
genre_mean_duration = df[df['duration_ms'] != -1].groupby('music_genre')['duration_ms'].mean()

# Заменить значения -1 на среднее значение для соответствующих жанров
for genre in genre_mean_duration.index:
    mean_duration = genre_mean_duration[genre]
    df.loc[(df['duration_ms'] == -1) & (df['music_genre'] == genre), 'duration_ms'] = mean_duration

# Заменить значения -1 в поле 'duration_ms' средним значением по всем данным
df.loc[df['duration_ms'] == -1, 'duration_ms'] = df['duration_ms'].mean()

# Определить количество строк, где 'duration_ms' равно -1 после замен
len(df[df['duration_ms'] == -1])



#Работа с признаками
st.write("Работа с признаками")


from collections import Counter

# Разделяем текст на слова и считаем их частоту для всех треков
words = ' '.join(df['track_name']).split()
word_counts = Counter(words)

# Выводим наиболее встречающиеся слова для всех треков
most_common_words = word_counts.most_common(100)
st.write("Самые часто встречающиеся слова:")
word_list = []
for word, count in most_common_words:
    word_list.append(word)
    st.write(f"{word}: {count}")
st.write(word_list)

# Определение уникальных музыкальных жанров
music_genres = ['Country', 'Rock', 'Alternative', 'Hip-Hop', 'Blues', 'Jazz',
       'Electronic', 'Anime', 'Rap', 'Classical']

# Анализ наиболее встречающихся слов для каждого музыкального жанра
for genre in music_genres:
    df_genre = df[df['music_genre'] == genre]

    words = ' '.join(df_genre['track_name']).split()
    word_counts = Counter(words)

    word_list = []  #list(word_counts.keys())

    # Выводим наиболее встречающиеся слова для каждого жанра
    most_common_words = word_counts.most_common(30)
    st.write(f"Самые часто встречающиеся слова в жанре {genre}:")
    for word, count in most_common_words:
        word_list.append(word)
        # print(f"{word}: {count}")
    st.write(word_list)

# Задаем списки слов для каждого жанра
classical = ['No.', 'Op.', 'Major,', 'Allegro', 'Minor,', 'Symphony', 'Act', 'I.', 'Piano', 'II.', 'BWV', 'Concerto', 'K.', 'Sonata', 'III.', 'Andante', 'Prelude']
electronic = ['Remix', 'Remix)', 'Mix', 'Edit', 'Original']
anime = ['Fantasy', '(From', 'Version', 'Theme', '"Final', 'Piano']
feat= ['(feat.']
love = ['Love', 'love', 'you', 'You', 'Me', 'me', 'Heart', 'heart', 'your', 'Your', 'My', 'my']

# Создаем список списков и соответствующий список названий столбцов
list_of_lists = [classical, electronic, love, anime, feat]
col_name = ['classical', 'electronic', 'love', 'anime', 'feat']

# Создаем флаги для каждого жанра
for i in range(len(list_of_lists)):
    words_to_check = list_of_lists[i]
    df[f'{col_name[i]}_flag'] = df['track_name'].apply(lambda x: 1 if any(word in x for word in words_to_check) else 0)



# Преобразование данных
df['mode'] = df['mode'].replace(['Major', 'Minor'], [1, 0])
df['duration_m'] = df['duration_ms'] / 60000
df['energy_dance_score'] = df['energy'] + df['danceability']
df['sound_intensity'] = df['loudness'] * df['tempo']
df['combined_liveness_valence'] = df['liveness'] + df['valence']
df['acoustic_instrumental_factor'] = df['acousticness'] * df['instrumentalness']
df['expressive_complexity'] = df['loudness'] + df['speechiness'] + df['tempo']
df['emotional_expression'] = df['speechiness'] + df['tempo']
df['liveliness'] = (df['liveness'] + df['acousticness']) / 2
df['mood_score'] = df['valence'] + df['tempo'] + df['loudness']
df['popularity_rating'] = df['danceability'] + df['energy'] + df['duration_ms']
df['all_features_sum'] = df['acousticness'] + df['danceability'] + df['energy'] + df['instrumentalness'] + \
                         df['liveness'] + df['loudness'] + df['speechiness'] + df['tempo'] + df['valence']
df['all_features_sum_per_sec'] = (df['acousticness'] + df['danceability'] + df['energy'] + df['instrumentalness'] + \
                                  df['liveness'] + df['loudness'] + df['speechiness'] + df['tempo'] + df['valence']) / \
                                 (df['duration_ms'] / 1000)

# Выводим результаты в Streamlit
st.write(df.head())



# Гистограмы новых факторов в разрезе жанров


# Список новых факторов
new_cols = ['mode', 'classical_flag', 'electronic_flag', 'love_flag', 'anime_flag', 'feat_flag', 'duration_m',
            'energy_dance_score', 'sound_intensity', 'combined_liveness_valence',
            'acoustic_instrumental_factor', 'expressive_complexity',
            'emotional_expression', 'liveliness', 'mood_score',
            'popularity_rating', 'all_features_sum', 'all_features_sum_per_sec']

# Для каждого нового фактора
for col_name in new_cols:
    st.write(col_name)
    df_check = df[['music_genre', f'{col_name}']]
    df_check = df_check.dropna(subset=['music_genre'])  # Убираем строки с пропущенными значениями в жанре

    # Создаем график
    fig, axes = plt.subplots(5, 2, figsize=(10, 10))

    # Перебираем каждый музыкальный жанр и строим гистограмму для фактора
    unique_genres = df_check['music_genre'].unique()
    for i, genre in enumerate(unique_genres):
        row = i % 5
        col = i // 5
        genre_data = df_check[df_check['music_genre'] == genre][f'{col_name}']
        genre_data.plot.hist(ax=axes[row, col])
        axes[row, col].set_title(genre)

    plt.tight_layout()
    st.pyplot(fig)
    st.write('-------------------------------------------------------------')

# Удаление столбца "track_name"
df = df.drop(columns='track_name')

# Подсчет количества значений в столбце "key"
key_counts = df['key'].value_counts()

# Преобразование фактора "key" в дамми-переменные
df = pd.get_dummies(df, columns=['key'], drop_first=True)

 # Отображение результата
st.write("Данные после удаления столбца 'track_name':")
st.write(df.head())
st.write("Количество значений в столбце 'key':")
st.write(key_counts)


# Выделение категориальных полей
categorical = list(df.select_dtypes(include=['object']).columns)
st.write("Категориальные поля:")
st.write(categorical)

# Выделение числовых полей
numerical = list(df.select_dtypes(include=['float64']).columns)
st.write("Числовые поля:")
st.write(numerical)





st.write("Выбор и обучение моделей")
#Выбор и обучение моделей
# Разделение данных на тренировочный и тестовый наборы
df_train = pd.DataFrame(df[pd.isna(df['music_genre'])==False])
df_test = pd.DataFrame(df[pd.isna(df['music_genre'])==True]).drop(columns='music_genre')

y = df_train["music_genre"]
X = df_train.drop(columns=["music_genre"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Стандартизация числовых значений
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical])
X_test_scaled = scaler.transform(X_test[numerical])

# Создание и обучение модели SVM
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)

# Предсказание жанра музыки
y_pred_svm = svm_model.predict(X_train_scaled)

# Оценка качества модели SVM
f1_svm = f1_score(y_train, y_pred_svm, average='micro')
st.write("f1_score модели SVM:", f1_svm)