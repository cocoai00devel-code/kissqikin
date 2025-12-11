import numpy as np
from scipy.io.wavfile import write

sample_rate = 8000
c = 35000  # 音速 cm/s

# 声道断面積と長さからフォルマント計算
def formants_from_vocal_tract(length_cm, area_factor=1.0):
    f1 = c / (4*length_cm) * area_factor
    f2 = 3*f1
    f3 = 5*f1
    return [f1,f2,f3]

# 母音ごとの声道長(cm)と断面積係数
vowel_tracts = {
    'A': (17,1.0),
    'E': (14,0.9),
    'I': (11,0.7),
    'O': (18,1.1),
    'U': (12,0.8)
}

# メロディ（簡易ループ可能）
melody_notes = ['C4','D4','E4','C4','E4','F4','G4','C4']
melody_freqs = {'C4':261.63,'D4':293.66,'E4':329.63,'F4':349.23,'G4':392.0}

# 音素ごとの基本長さ（秒）
base_durations = {
    'vowel': 0.25,
    'consonant': 0.1,
    'space': 0.1
}

# 子音雑音生成
def consonant_wave(duration):
    t = np.linspace(0,duration,int(sample_rate*duration),False)
    return np.random.uniform(-0.4,0.4,t)*np.exp(-5*t)

# 歌詞全体
lyrics_text = """
Daisy Bell, Daisy Bell, I love you,
Please, won't you tell me how you'd like to be sung to?
""".replace('\n',' ')

# 音素化
def text_to_phonemes(text):
    phonemes = []
    for c in text.upper():
        if c.isalpha():
            phonemes.append(c)
        elif c==' ' or c==',':
            phonemes.append(' ')
    return phonemes

phonemes = text_to_phonemes(lyrics_text)

# 母音強弱（アクセント簡易モデル: 母音の位置で強くする）
def vowel_volume(index, total):
    # 前半は小さめ、フレーズの末尾は大きく
    return 0.5 + 0.5*(index/total)

# 音素→波形生成（タイミング・強弱対応）
def phoneme_wave(note, phoneme, index, total):
    if phoneme==' ':
        dur = base_durations['space']
        return np.zeros(int(sample_rate*dur))
    elif phoneme in vowel_tracts:
        dur = base_durations['vowel']
        length, area = vowel_tracts[phoneme]
        formants = formants_from_vocal_tract(length, area)
        f_note = melody_freqs[note]
        bend = np.linspace(0,10,int(sample_rate*dur))
        wave = np.zeros(int(sample_rate*dur))
        for f in formants:
            wave += (0.45/len(formants)) * np.sin(2*np.pi*(f_note+f+bend)*np.linspace(0,dur,int(sample_rate*dur),False))
        # アタック・デケイ + 母音強弱
        env = np.linspace(0,1,len(wave)//4)
        env = np.concatenate([env,np.ones(len(wave)-len(env))])
        wave *= env * vowel_volume(index,total)
        return wave
    else:
        dur = base_durations['consonant']
        return consonant_wave(dur)

# 曲全体生成
song = np.array([],dtype=np.float32)
total_phonemes = len(phonemes)
for i,p in enumerate(phonemes):
    note = melody_notes[i % len(melody_notes)]
    song = np.concatenate((song, phoneme_wave(note,p,i,total_phonemes)))

# 保存
song_int16 = np.int16(song*32767)
write('daisy_bell_full_realtime_vocal.wav',sample_rate,song_int16)

print("daisy_bell_full_realtime_vocal.wav を生成しました。")
これをjsの音声強瀬に呼び出してほしい