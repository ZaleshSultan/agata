import pyttsx3

engine = pyttsx3.init()

for i, voice in enumerate(engine.getProperty("voices")):
    print("=" * 60)
    print("INDEX:", i)
    print("NAME:", voice.name)
    print("ID:", voice.id)
    print("LANGUAGES:", voice.languages)