#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import sys

print(sys.executable)

import pkg_resources

installed_packages = pkg_resources.working_set
installed_packages_list = sorted(
    ["%s==%s" % (i.key, i.version) for i in installed_packages]
)

for package in installed_packages_list:
    print(package)

from pathlib import Path

from vllm import LLM, SamplingParams

eng_prompt = "For the given English sentence, restore the upper-case characters (if applicable) and add punctuation WITHOUT CHANGING ANY WORDS. Answer in English without any explanation. \nHere is the sentence: {}\nHere is the output:"
zho_prompt = (
    "为以下中文句子添加标点符号，但不能改变文字本身。以中文回答：{}\n这是回答："
)
deu_prompt = "Stellen Sie für den gegebenen deutschen Satz die Zeichen der Oberklasse wieder her (falls zutreffend) und fügen Sie Satzzeichen hinzu, OHNE JEGLICHE WÖRTER ZU ÄNDERN. Antwort auf Deutsch ohne Erklärung. \nHier ist der Satz: {}\nHier ist die Ausgabe:"
fra_prompt = "Pour la phrase française donnée, restaurez les caractères de classe supérieure (le cas échéant) et ajoutez de la ponctuation SANS CHANGER AUCUN MOT. Réponse en français sans aucune explication.\nVoici la phrase: {}\nVoici le résultat:"
spa_prompt = "Para la oración en español dada, restaure los caracteres de clase alta (si corresponde) y agregue puntuación SIN CAMBIAR NINGUNA PALABRA. Responde en español sin ninguna explicación. \nAquí está la oración: {}\nAquí está el resultado:"
ita_prompt = "Per la frase italiana data, ripristina i caratteri di classe superiore (se applicabile) e aggiungi la punteggiatura SENZA CAMBIARE NESSUNA PAROLA. Rispondi in italiano senza alcuna spiegazione. \nEcco la frase: {}\nEcco l'output:"
nld_prompt = "Stellen Sie für den angegebenen niederländischen Satz die Zeichen der oberen Klasse wieder her (falls zutreffend) und fügen Sie Satzzeichen hinzu, OHNE JEGLICHE WÖRTER ZU ÄNDERN. Antworten Sie auf Niederländisch ohne Erklärung. \nHier ist der Satz: {}\nHier ist die Ausgabe:"
por_prompt = "Para a frase em português fornecida, restaure os caracteres de classe alta (se aplicável) e adicione pontuação SEM ALTERAR NENHUMA PALAVRA. Responda em português sem nenhuma explicação. \nAqui está a frase: {}\nAqui está o resultado:"
pol_prompt = "Dla podanego polskiego zdania przywróć znaki wyższej klasy (jeśli dotyczy) i dodaj znaki interpunkcyjne BEZ ZMIANY ŻADNYCH SŁÓW. Odpowiadaj po polsku bez żadnych wyjaśnień. \nOto zdanie: {}\nOto wynik:"


# only rich-resource languages. see https://arxiv.org/pdf/2401.04088.pdf
prompt_map = {
    "eng": eng_prompt,
    "zho": zho_prompt,
    "fra": fra_prompt,
    "deu": deu_prompt,
    "ita": ita_prompt,
    "spa": spa_prompt,
    "nld": nld_prompt,
    "por": por_prompt,
    "pol": pol_prompt,
}


class MixtralPunctuator(object):
    def __init__(self):

        self.model = LLM("mistralai/Mistral-7B-Instruct-v0.1")
        self.config = SamplingParams(top_k=1, max_tokens=500)

    def process_batch(self, batch):
        # batch format: [uttid, lang_id, content]
        prompts = [prompt_map[lang_id].format(content) for _, lang_id, content in batch]
        outputs = self.model.generate(prompts, self.config)

        ans = [output.outputs[0].text.replace("\n", "") for output in outputs]
        ans = [(uttid, text) for (uttid, _, _), text in zip(batch, ans)]

        return ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=Path, help="Input owsm text file")
    parser.add_argument("--output", "-o", type=Path, help="Output punctuated file")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size to submit jobs to LLM engine. Larger batch size"
        "help the engine to better coordinate the resources",
    )
    parser.add_argument(
        "--lang", default="eng", choices=list(prompt_map.keys()), help="input language"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level="INFO",
        format=f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # (1) read and parse lines
    data = {}
    for line in open(args.input, encoding="utf-8"):
        uttid, content = line.strip().split(maxsplit=1)
        data[uttid] = content
    logging.info(f"Total input examples: {len(data)}")

    # (4) Init Punctuator
    punctuator = MixtralPunctuator()

    # (5) Start the job
    writer = open(args.output, "w", encoding="utf-8")
    data = [(k, args.lang, v) for k, v in data.items()]
    bidx = 0
    while bidx * args.batch_size < len(data):
        batch = data[
            bidx * args.batch_size : min((bidx + 1) * args.batch_size, len(data))
        ]
        results = punctuator.process_batch(batch)
        for uttid, content in results:
            writer.write(f"{uttid} {content}\n")
        writer.flush()

        bidx += 1


if __name__ == "__main__":
    main()
