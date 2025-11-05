import soundfile
from espnet2.bin.tts_inference import Text2Speech

# 事前学習済みの日本語音声合成モデルを読み込み
text2speech = Text2Speech.from_pretrained("kan-bayashi/jsut_vits_prosody")

# 音声合成したいテキスト
text = "パソナはお客様の課題に寄り添い、メソッドとテクノロジーを掛け合わせることで、お客様の状況にあった幅広い提案ができます。"

# 音声合成を実行
speech = text2speech(text)["wav"]

# 音声ファイルを保存
soundfile.write("output.wav", speech.numpy(), text2speech.fs, "PCM_16")
