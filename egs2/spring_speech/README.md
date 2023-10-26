# SPRING_INX_ESPnet_Recipe
ESPnet Recipes to get started with SPRING-INX data.

## Data Statistics

We are releasing the first set of valuable data amounting to 2000 hours (both Audio and corresponding manually transcribed data) which was collected, cleaned and prepared for ASR system building in 10 Indian languages such as Assamese, Bengali Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi and Tamil in the public domain.

The prepared language-wise dataset was then split to train, valid and test tests. The number hours of training and validation data per language is presented in following table.

| Language  | Train | Valid | Test | Total (Approx.) |
|-----------|-------|-------|------|-----------------|
| Assamese  | 50.6  | 5.1   | 5.0  | 61              |
| Bengali   | 374.7 | 40.0  | 5.0  | 420             |
| Gujarati  | 175.5 | 19.6  | 5.0  | 200             |
| Hindi     | 316.4 | 29.7  | 5.0  | 351             |
| Kannada   | 82.5  | 9.7   | 4.8  | 97              |
| Malayalam | 214.7 | 24.7  | 5.0  | 245             |
| Marathi   | 130.4 | 14.4  | 5.2  | 150             |
| Odia      | 82.5  | 9.3   | 4.7  | 96              |
| Punjabi   | 140.0 | 15.1  | 5.1  | 159             |
| Tamil     | 200.7 | 20.0  | 5.1  | 226             |

For more information you can refer the arVix paper : https://arxiv.org/abs/2310.14654v2

## Usage

 You can select the language you want by specifying the "lang" parameter in the run.sh file. By default, it is set to "assamese".

