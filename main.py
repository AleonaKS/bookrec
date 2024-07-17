
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Dropout, concatenate
#
#
# num_books = len(books_df['isbn'].unique())
# num_users = len(reactions_df['user_id'].unique())
#
# # Входы для модели
# book_input = Input(shape=(1,), name='Book_Input')
# user_input = Input(shape=(1,), name='User_Input')
#
# # Эмбеддинги для книг и пользователей
# book_embedding = Embedding(input_dim=num_books, output_dim=50, input_length=1, name='Book_Embedding')(book_input)
# user_embedding = Embedding(input_dim=num_users, output_dim=50, input_length=1, name='User_Embedding')(user_input)
#
# # Сглаживание эмбеддингов
# book_vec = Flatten(name='Flatten_Books')(book_embedding)
# user_vec = Flatten(name='Flatten_Users')(user_embedding)
#
# # Конкатенация эмбеддингов
# concat = concatenate([book_vec, user_vec], name='Concatenate')
#
# # Полносвязные слои
# dense = Dense(128, activation='relu', name='Fully_Connected')(concat)
# dropout = Dropout(0.5, name='Dropout')(dense)
# output = Dense(1, activation='sigmoid', name='Output')(dropout)
#
# # Создание модели
# model = Model(inputs=[book_input, user_input], outputs=output)
#
# # Компиляция модели
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# # Вывод структуры модели
# model.summary()
#
#
# df = pd.merge(reviews_df, books_df, on = 'isbn')
#
# df = df[['isbn', 'rating', 'user_id']]
#
# train, test = train_test_split(df, test_size = 0.2, random_state = 0)
#
# model.fit([train['user_id'], train['isbn']], train['rating'], epochs = 3, batch_size = 512, verbose = 1)
