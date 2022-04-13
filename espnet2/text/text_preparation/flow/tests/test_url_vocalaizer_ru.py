import unittest

from text_preparation.flow.mutators.ru import URLVocalizer
from text_preparation.flow.tests.fixtures import url_data


class URLVocalizerTest(unittest.TestCase):

    def test_urls_numbers(self):
        for test, expected in url_data.items():
            actual = URLVocalizer()(test)
            self.assertEqual(expected, actual)
