import re
import inflect
import argparse

# Initialize the inflect engine for number to word conversion
p = inflect.engine()

def remove_text_between_markers(text):
    # Remove text between {}, <>, [], **, (), and +
    text = re.sub(r'\{.*?\}', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\+.*?\+', '', text)
    
    # Remove nested parentheses
    while re.search(r'\([^()]*\)', text):
        text = re.sub(r'\([^()]*\)', '', text)
    
    
    return text

def convert_numbers_to_words(text):
    # Convert all numbers to words
    return re.sub(r'\d+', lambda x: p.number_to_words(x.group()), text)

def process_text(text):
    # Remove text between markers
    text = remove_text_between_markers(text)
    # Convert numbers to words
    text = convert_numbers_to_words(text)
    # Replace & with and
    text = text.replace('&', ' and ')
    # Remove underscores and hyphens
    text = text.replace('_', '').replace('-', '')
    text = text.replace('–', '')
    text = text.replace('*', '')

    text = text.replace('/', ' slash ')
    text = text.replace('’', "'")
    text = text.replace('‘', "'")
    text = text.replace("'EM", 'em')
    text = text.replace("THEY 'RE", "THEY'RE")
    text = text.replace("I 'M", "I'M")
    text = text.replace("I 'VE", "I'VE")
    text = text.replace("I 'LL", "I'LL")
    text = text.replace("I 'D", "I'D")
    text = text.replace("YOU 'RE", "YOU'RE")
    text = text.replace("THEY 'VE", "THEY'VE")
    text = text.replace("THEY 'LL", "THEY'LL")
    text = text.replace("THEY 'D", "THEY'D")
    text = text.replace("HE 'S", "HE'S")
    text = text.replace("'CAUSE", "CAUSE")
    text = text.replace(',', '')
    text = text.replace('.', '')
    text = text.replace('?', '')
    text = text.replace('!', '')
    text = text.replace(';', '')
    text = text.replace('…', '')
    text = text.replace('+UM', '')
    text = text.replace('+UH', '')
    text = text.replace('+', '')
    text = text.replace(':', '')

    text = text.upper()

    # Remove multiple whitespace with a single space
    text = re.sub(r'\s+', ' ', text)

    text = text.strip()
    return text

def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    with open(output_path, 'w', encoding='utf-8') as file:
        for line in lines:
            # Split into utt_id and text
            utt_id, text = line.split(maxsplit=1)
            # Process the text
            processed_text = process_text(text)
            # Write the processed line to the output file
            file.write(f"{utt_id} {processed_text}\n")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess text file')
    parser.add_argument('input_path', type=str, help='Path to the input text file')
    parser.add_argument('output_path', type=str, help='Path to the output text file')

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    process_file(input_path, output_path)