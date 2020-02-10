from __future__ import print_function, division
from builtins import range, input
import pygame
import time
#import speech_recognition as sr
import os, sys
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, \
    Bidirectional, RepeatVector, Concatenate, Activation, Dot, Lambda
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
import time
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from difflib import get_close_matches


#import socket
#live demo

import glob

import random

import numpy as np
import pandas as pd

import cv2

from timer import Timer
from frame import video2frames, images_normalize, frames_downsample, images_crop
from frame import images_resize_aspectratio, frames_show, frames2files, files2frames, video_length
from videocapture import video_start, frame_show, video_show, video_capture
from opticalflow import frames2flows, flows2colorimages, flows2file, flows_add_third_channel
from datagenerator import VideoClasses
from model_i3d import I3D_load
from predict import probability2label
#end lib for demo
if len(K.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM
    from keras.layers import CuDNNGRU as GRU

BATCH_SIZE = 64
EPOCHS = 200
LATENT_DIM = 256
LATENT_DIM_DECODER = 256
NUM_SAMPLES = 150
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

pygame.init()

display_width = 800
display_height = 600

black = (0,0,0)
white = (255,255,255)
gray = (180,180,180)


gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('ASL')

gameDisplay.fill(white)
carImg = pygame.image.load('img.jpg')
gameDisplay.blit(carImg,(0,0))

def close():
    pygame.quit()
    quit()


def message_display(text):
    largeText = pygame.font.Font('freesansbold.ttf',30)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.center = ((400),(100))
    gameDisplay.blit(TextSurf, TextRect)

    pygame.display.update()


def text_objects(text, font):
    textSurface = font.render(text, True, black)
    return textSurface, textSurface.get_rect()

def button(msg,x,y,w,h,ic,ac,action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    if x+w > mouse[0] > x and y+h > mouse[1] > y:
        pygame.draw.rect(gameDisplay, ac,(x,y,w,h))

        if click[0] == 1 and action != None:
            action()
    else:
        pygame.draw.rect(gameDisplay, ic,(x,y,w,h))

    smallText = pygame.font.SysFont("comicsansms",20)
    textSurf, textRect = text_objects(msg, smallText)
    textRect.center = ( (x+(w/2)), (y+(h/2)) )
    gameDisplay.blit(textSurf, textRect)

def softmax_over_time(x):
    assert (K.ndim(x) > 2)
    e = K.exp(x - K.max(x, axis=1, keepdims=True))
    s = K.sum(e, axis=1, keepdims=True)
    return e / s



# Where we will store the data
input_texts = []  # sentence in original language
target_texts = []  # sentence in target language
target_texts_inputs = []  # sentence in target language offset by 1


def stack_and_transpose(x):
    # x is a list of length T, each element is a batch_size x output_vocab_size tensor
    x = K.stack(x)  # is now T x batch_size x output_vocab_size tensor
    x = K.permute_dimensions(x, pattern=(1, 0, 2))  # is now batch_size x T x output_vocab_size
    return x


def one_step_attention(h, st_1):
    st_1 = attn_repeat_layer(st_1)
    x = attn_concat_layer([h, st_1])
    x = attn_dense1(x)
    alphas = attn_dense2(x)
    context = attn_dot([alphas, h])

    return context


def decode_sequence(input_seq):
    enc_out = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs['<sos>']
    eos = word2idx_outputs['<eos>']

    s = np.zeros((1, LATENT_DIM_DECODER))
    c = np.zeros((1, LATENT_DIM_DECODER))

    # Create the translation
    output_sentence = []
    for j in range(max_len_target):
        o, s, c = decoder_model.predict([target_seq, enc_out, s, c])
        # Get next word
        idx = np.argmax(o.flatten())
        # End sentence of EOS
        if eos == idx:
            break

        word = ''
        if idx > 0:
            word = idx2word_trans[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx

    return (' '.join(output_sentence))


t = 0
for line in open('final.txt'):
        t += 1
        if t > NUM_SAMPLES:
            break

        if '\t' not in line:
            continue

        input_text, translation = line.rstrip().split('\t')
        target_text = translation + ' <eos>'
        target_text_input = '<sos> ' + translation
        input_texts.append(input_text)
        target_texts.append(target_text)
        target_texts_inputs.append(target_text_input)
        # only keep a limited number of samples



tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)
word2idx_inputs = tokenizer_inputs.word_index
#print('Found %s unique input tokens.' % len(word2idx_inputs))
max_len_input = max(len(s) for s in input_sequences)
tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs)
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)
word2idx_outputs = tokenizer_outputs.word_index
#print('Found %s unique output tokens.' % len(word2idx_outputs))
num_words_output = len(word2idx_outputs) + 1
max_len_target = max(len(s) for s in target_sequences)
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)
decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=max_len_target, padding='post')
decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')

print('Loading word vectors...')
word2vec = {}
with open(os.path.join('glove.6B.%sd.txt' % EMBEDDING_DIM),encoding="utf8") as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:])
    word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))

def rnn():
    # prepare embedding matrix
    print('Filling pre-trained embeddings...')
    num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word2idx_inputs.items():
        if i < MAX_NUM_WORDS:
            embedding_vector = word2vec.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all zeros.
                embedding_matrix[i] = embedding_vector

    # create embedding layer
    embedding_layer = Embedding(
        num_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=max_len_input,
        # trainable=True
    )
    return embedding_layer

    # create embedding layer



decoder_targets_one_hot = np.zeros(
        (
            len(input_texts),
            max_len_target,
            num_words_output
        ),
        dtype='float32'

    )
for i, d in enumerate(decoder_targets):
        for t, word in enumerate(d):
            decoder_targets_one_hot[i, t, word] = 1

encoder_inputs_placeholder = Input(shape=(max_len_input,))
emd=rnn()
x = emd(encoder_inputs_placeholder)
encoder = Bidirectional(LSTM(
        LATENT_DIM,
        return_sequences=True,
        # dropout=0.5 # dropout not available on gpu
    ))


encoder_outputs = encoder(x)
decoder_inputs_placeholder = Input(shape=(max_len_target,))
    # this word embedding will not use pre-trained vectors
    # although you could
decoder_embedding = Embedding(num_words_output, EMBEDDING_DIM)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)
attn_repeat_layer = RepeatVector(max_len_input)
attn_concat_layer = Concatenate(axis=-1)
attn_dense1 = Dense(10, activation='tanh')
attn_dense2 = Dense(1, activation=softmax_over_time)
attn_dot = Dot(axes=1)  # to perform the weighted sum of alpha[t] * h[t]
# define the rest of the decoder (after attention)
decoder_lstm = LSTM(LATENT_DIM_DECODER, return_state=True)
decoder_dense = Dense(num_words_output, activation='softmax')
initial_s = Input(shape=(LATENT_DIM_DECODER,), name='s0')
initial_c = Input(shape=(LATENT_DIM_DECODER,), name='c0')
context_last_word_concat_layer = Concatenate(axis=2)
s = initial_s
c = initial_c

outputs = []
for t in range(max_len_target):  # Ty times
    # get the context using attention
    context = one_step_attention(encoder_outputs, s)
    # we need a different layer for each time step
    selector = Lambda(lambda x: x[:, t:t + 1])
    xt = selector(decoder_inputs_x)
    # combine
    decoder_lstm_input = context_last_word_concat_layer([context, xt])
    o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[s, c])
    # final dense layer to get next word prediction
    decoder_outputs = decoder_dense(o)
    outputs.append(decoder_outputs)


stacker = Lambda(stack_and_transpose)
outputs = stacker(outputs)

model = Model(
        inputs=[
            encoder_inputs_placeholder,
            decoder_inputs_placeholder,
            initial_s,
            initial_c,
        ],
        outputs=outputs
    )

# compile the model
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
#z = np.zeros((NUM_SAMPLES, LATENT_DIM_DECODER)) # initial [s, c]
#r = model.fit(
 # [encoder_inputs, decoder_inputs, z, z], decoder_targets_one_hot,
  #batch_size=BATCH_SIZE,
  #epochs=EPOCHS,
  #validation_split=0.2





encoder_model = Model(encoder_inputs_placeholder, encoder_outputs)
encoder_outputs_as_input = Input(shape=(max_len_input, LATENT_DIM * 2,))
decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)
context = one_step_attention(encoder_outputs_as_input, initial_s)
decoder_lstm_input = context_last_word_concat_layer([context, decoder_inputs_single_x])
# lstm and final dense
o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[initial_s, initial_c])
decoder_outputs = decoder_dense(o)


decoder_model = Model(
        inputs=[
            decoder_inputs_single,
            encoder_outputs_as_input,
            initial_s,
            initial_c
        ],
        outputs=[decoder_outputs, s, c]
    )

idx2word_eng = {v: k for k, v in word2idx_inputs.items()}
idx2word_trans = {v: k for k, v in word2idx_outputs.items()}
model.load_weights('final.h5')




#print("num samples:", len(input_texts))
    # make it a layer
    # compile the model
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # train the model
    # z = np.zeros((NUM_SAMPLES, LATENT_DIM_DECODER))  # initial [s, c]
    # r = model.fit(
    # [encoder_inputs, decoder_inputs, z, z], decoder_targets_one_hot,
    # batch_size=BATCH_SIZE,
    # epochs=EPOCHS,
    # validation_split=0.2)
       # create the model object
    # map indexes back into real words# so we can view the results


diVideoSet = {"sName": "chalearn",
              "nClasses": 20,  # number of classes
              "nFramesNorm": 100,  # number of frames per video
              "nMinDim": 240,  # smaller dimension of saved video-frames
              "tuShape": (224, 226),  # height, width
              "nFpsAvg": 10,
              "nFramesAvg": 50,
              "fDurationAvg": 5.0}  # seconds

# files
sClassFile = "class.csv"

print("\nStarting gesture recognition live demo ... ")
print(os.getcwd())
print(diVideoSet)

# load label description
oClasses = VideoClasses(sClassFile)

sModelFile = "epochs_001-val_acc_0.980.hdf5"
h, w = 224, 224
keI3D = I3D_load(sModelFile, diVideoSet["nFramesNorm"], (h, w, 2), oClasses.nClasses)
def live ():
    gameDisplay.blit(carImg,(0,0))



# open a pointer to the webcam video stream
    oStream = video_start(device=1, tuResolution=(320, 240), nFramePerSecond=diVideoSet["nFpsAvg"])


    timer = Timer()
    sResults = ""
    nCount=0
    while True:
        # show live video and wait for key stroke
        key = video_show(oStream, "green", "Press <blank> to start", sResults, tuRectangle=(h, w))

        # start!
        if key == ord(' '):
            # countdown n sec
            video_show(oStream, "orange", "Recording starts in ", tuRectangle=(h, w), nCountdown=3)

            # record video for n sec
            fElapsed, arFrames, _ = video_capture(oStream, "red", "Recording ", \
                                                  tuRectangle=(h, w), nTimeDuration=int(diVideoSet["fDurationAvg"]),
                                                  bOpticalFlow=False)
            print("\nCaptured video: %.1f sec, %s, %.1f fps" % \
                  (fElapsed, str(arFrames.shape), len(arFrames) / fElapsed))

            # show orange wait box
            frame_show(oStream, "orange", "Translating sign ...", tuRectangle=(h, w))

            # crop and downsample frames
            arFrames = images_crop(arFrames, h, w)
            arFrames = frames_downsample(arFrames, diVideoSet["nFramesNorm"])

            # Translate frames to flows - these are already scaled between [-1.0, 1.0]
            print("Calculate optical flow on %d frames ..." % len(arFrames))
            timer.start()
            arFlows = frames2flows(arFrames, bThirdChannel=False, bShow=True)
            print("Optical flow per frame: %.3f" % (timer.stop() / len(arFrames)))

            # predict video from flows
            print("Predict video with %s ..." % (keI3D.name))
            arX = np.expand_dims(arFlows, axis=0)
            arProbas = keI3D.predict(arX, verbose=1)[0]
            nLabel, sLabel, fProba = probability2label(arProbas, oClasses, nTop=3)

            sResults = "Sign: %s (%.0f%%)" % (sLabel, fProba * 100.)
            print(sResults)
            nCount += 1

        # quit
            break

    # do a bit of cleanup
    message_display(sResults)
    oStream.release()
    cv2.destroyAllWindows()


	# dataset










	# loop over action states








def rec():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak Anything :")
        audio = r.listen(source)
        print("done")
    try:
        text = r.recognize_google(audio)

        print("You said :"  , text)
         #   print("You said : {}".format(text))
         #print(r.recognize_google(audio))
    except:
        print("Sorry could not recognize what you said")


    return text



def s2t():
    gameDisplay.blit(carImg,(0,0))

    while True:
        text = rec()
        axs = list(get_close_matches(text, input_texts))
        print(axs)
        text=axs[0]
        for i in range(len(input_texts)):
            if text == input_texts[i]:
                input_seq = encoder_inputs[i:i + 1]
                translation = decode_sequence(input_seq)
                print('Input sentence:', input_texts[i])
                print('Predicted translation:', translation)
                UDP_IP = "127.0.0.1"
                UDP_PORT = 5065
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.sendto((translation).encode(), (UDP_IP, UDP_PORT))
    #message_display(translation)    #
        #ans = input("Continue? [Y/n]")
        #if ans and ans.lower().startswith('n'):
        break

    message_display( translation)

def main():

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        button("Speak!",150,450,100,50,gray,gray,s2t)
        button("live", 550, 450, 100, 50, gray, gray, live)

        pygame.display.update()


if __name__ == '__main__':
    main()
