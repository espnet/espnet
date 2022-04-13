import unittest
from text_preparation.framing_numbers.tests.fixtures import test_cases, test_texts
from text_preparation.framing_numbers.framing_numbers import NumberWithSpacesCase


class FramingNumbersTest(unittest.TestCase):

    def test_cases(self):
        for text, expected in test_cases:
            match_pos = NumberWithSpacesCase().get_match_position(text)
            if expected is None:
                self.assertIsNone(match_pos)
            else:
                match = text[match_pos[0]:match_pos[1]]
                self.assertEqual(match, expected)

    def test_texts(self):
        for i, (text, expected) in enumerate(test_texts.items()):
            result = NumberWithSpacesCase().frame_numbers(text)
            self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
