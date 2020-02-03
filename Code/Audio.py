import numpy as np
import librosa
import matplotlib.pyplot as plt
import noisereduce as nr

def plotAudio(output):
    fig, ax = plt.subplots(nrows=1,ncols=1) #, figsize=(30,10))
    plt.plot(output, color='blue')
    ax.set_xlim((0, len(output)))
    ax.margins(2, -0.1)
    plt.show()

def plotAudio2(output):
    fig, ax = plt.subplots(nrows=1,ncols=1) #, figsize=(20,4))
    plt.plot(output, color='blue')
    ax.set_xlim((0, len(output)))
    plt.show()

for i in range(1, 11):
    print(i)
    raw_audio, sr = librosa.load('../Data/Stop/'+str(i)+'.wav')

    noisy_part = raw_audio[0:50000]  # Empherically selected noisy_part position for every sample
    nr_audio = nr.reduce_noise(audio_clip=raw_audio, noise_clip=noisy_part, verbose=False)

    #plotAudio(nr_audio)

    #plotAudio2(nr_audio)

    fig,ax = plt.subplots(1)
    pxx, freqs, bins, im = ax.specgram(x=nr_audio, Fs=18000, noverlap=511, NFFT=512)
    fig.canvas.draw()
    #plt.show(fig)
    plt.savefig('../Images/Stop/'+str(i)+'.png')
