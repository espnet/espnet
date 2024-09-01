# This is ESPnet2 Speech Language Model (SpeechLM) Recipe
🎙️ ``Task``: Singing Voice Synthesis (SVS)

📊 ``Corpus``: Mandarin Single Female Singer Corpus, [Opencpop](https://wenet.org.cn/opencpop/download/).

## Before Running
* Download this corpus from the [official websites](https://wenet.org.cn/opencpop/download/)

* Organize the data as
    ```
    opencpop/speechlm1
        |__downloads
        |   |__raw_data____midis
        |   |           |__textgrids
        |   |           |__wavs
        |   |__segments____wavs
        |               |__test.txt
        |               |__train.txt
        |               |__transcriptions.txt
        |__local
            |__midi-note.scp
    ```

## Data Structure
* We use ``label`` as the conditional entry, ulter it to triplet using  ``text`` and ``score`` in the original SVS. ``wav.scp`` is the target entry.
