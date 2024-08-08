from typing import Union
from gensim.models import FastText, Word2Vec, KeyedVectors
from typing import List
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import regex as re
import pickle

from fastapi import FastAPI


modelvector = Word2Vec.load('w2v_OA_CR_100d.bin')

app = FastAPI()
def file_to_sentence_list(file_path):
    with open(file_path, 'r') as file:
        text = file.read()

    # Splitating the text into sentences using
    # delimiters like '.', '?', and '!'
    sentences = [sentence.strip() for sentence in re.split(
        r'(?<=[.!?])\s+', text) if sentence.strip()]

    return sentences

file_path = 'descriptions.txt'
text_data = file_to_sentence_list(file_path)

# --- Model Saving and Loading ---
model_filename = 'chat_model.keras'
tokenizer_filename = 'tokenizer.pickle'

try:
    # Try to load the model and tokenizer
    model = tf.keras.models.load_model(model_filename)
    with open(tokenizer_filename, 'rb') as handle:
        tokenizer = pickle.load(handle,allow_pickle=True)

    # Get total_words and max_sequence_len from the loaded tokenizer
    total_words = len(tokenizer.word_index) + 1
    max_sequence_len = model.layers[0].input_length + 1  # Add 1 because we removed the last word for y

    print("Model and tokenizer loaded successfully.")

except:
    # If the files don't exist, train the model and save it
    print("Model and tokenizer not found. Training a new model...")

    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_data)
    total_words = len(tokenizer.word_index) + 1

    # Create input sequences
    input_sequences = []
    for line in text_data:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    # Pad sequences and split into predictors and label
    max_sequence_len = max([len(seq) for seq in input_sequences])
    input_sequences = np.array(pad_sequences(
        input_sequences, maxlen=max_sequence_len, padding='pre'))
    X, y = input_sequences[:, :-1], input_sequences[:, -1]

    # Convert target data to one-hot encoding
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    # Define the model
    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=max_sequence_len - 1))
    model.add(LSTM(128))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X, y, epochs=500, verbose=1)

    # Save the model and tokenizer
    model.save(model_filename)
    with open(tokenizer_filename, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Model and tokenizer saved.")

# --- Generate next word predictions (using loaded or newly trained model) ---


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/word_similarity/{word}/{top}")
async def word_similarity(word: str, top: int) -> List[str]:
    """
    Finds the most similar words to a given word using a Word2Vec model.
    """
    try:
        similar_words = [word for word, similarity in modelvector.wv.most_similar(word, topn=top)]
        return similar_words
    except KeyError:
        return {"error": f"Word '{word}' not found in the vocabulary."}


@app.get("/auto_complete/{text}/{next}")
async def auto_complete(text: str, next: int) -> str:
    """
    Completes the given text by predicting the next words using a language model.
    """
    seed_text = text
    next_words = next

    try:
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
            predicted_probs = model.predict(token_list)
            predicted_id = np.argmax(predicted_probs)
            # Handle cases where the predicted ID is not in the vocabulary
            predicted_word = tokenizer.index_word.get(predicted_id, "<UNK>")
            seed_text += " " + predicted_word

        return seed_text

    except Exception as e:
        return f"Error: {e}"
