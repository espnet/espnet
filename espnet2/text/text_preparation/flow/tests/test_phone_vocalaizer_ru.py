import unittest

from text_preparation.flow.mutators.ru import PhoneVocalizer
from text_preparation.flow.tests.fixtures import phone_data


class PhoneVocalizerTest(unittest.TestCase):

    def test_russian_numbers(self):
        for test, expected in phone_data.get('ru', dict()).items():
            actual = PhoneVocalizer()(test)
            self.assertEqual(expected, actual)

    def test_ukraine_numbers(self):
        for test, expected in phone_data.get('ua', dict()).items():
            actual = PhoneVocalizer()(test)
            self.assertEqual(expected, actual)

    def test_kazakhstan_numbers(self):
        for test, expected in phone_data.get('kz', dict()).items():
            actual = PhoneVocalizer()(test)
            self.assertEqual(expected, actual)

    def test_belarus_numbers(self):
        for test, expected in phone_data.get('br', dict()).items():
            actual = PhoneVocalizer()(test)
            self.assertEqual(expected, actual)

    def test_unprocessed_numbers(self):
        for test, expected in phone_data.get('unprocessed', dict()).items():
            actual = PhoneVocalizer()(test)
            self.assertEqual(expected, actual)