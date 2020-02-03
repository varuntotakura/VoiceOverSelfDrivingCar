import pyaudio
import wave

print('#            #            Right              #           #')
for j in range(1,11):
    print(str(j))
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 2
    WAVE_OUTPUT_FILENAME = "../Data/Right/"
     
    audio = pyaudio.PyAudio()
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    ##    print("recording...",str(j))
    frames = []
     
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    ##    print("finished recording",str(j))
     
     
    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    fil = WAVE_OUTPUT_FILENAME+str(j)+'.wav'
    waveFile = wave.open(fil, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
