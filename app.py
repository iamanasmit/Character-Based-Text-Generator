import streamlit as st
import tensorflow as tf

import pickle
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('model.pickle', 'rb') as handle:
    model = pickle.load(handle)

num_tokens = len(tokenizer.word_index) + 1
input_timesteps = 100
def generate_text(model, tokenizer, seed, num_chars, temperature=1.0):
    text=seed
    for _ in range(num_chars):
        text_input_seq=tokenizer.texts_to_sequences([text[-input_timesteps:]])[0]
        text_input_one_hot=tf.one_hot(text_input_seq, num_tokens)

        pred=model.predict(tf.expand_dims(text_input_one_hot, axis=0))[0, -1, :]#we only want the last character
        preds=tf.math.log(pred)/temperature

        next_char=tf.random.categorical(tf.expand_dims(preds, axis=0), num_samples=1)
        next_char=tokenizer.sequences_to_texts([next_char.numpy()][0])[0]

        text+=next_char
    
    return text

st.title('Art of War')
st.write('This is a text generation model using RNNs. It is trained on a dataset of 50k lines of the classic novel "The Art of War" by Sun Tzu.')

seed = st.text_input('Enter a seed text', '1. Sun Tzŭ said: The art of war is of vital importance to the State.')
num_chars = st.number_input('Number of characters to generate', min_value=1, max_value=1000, value=200)
temperature = st.slider('Temperature', min_value=0.1, max_value=2.0, value=1.0)

if st.button('Generate'):
    generated_text = generate_text(model, tokenizer, seed, num_chars, temperature)
    st.write(generated_text)

