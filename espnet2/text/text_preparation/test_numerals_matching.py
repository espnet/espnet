import sys
import unittest

try:
    from text_preparation.numerals_matching import NumeralsMatching_ru
except ModuleNotFoundError:
    from numerals_matching import NumeralsMatching_ru


class TestNumeralsMatchingRu(unittest.TestCase):
    num_matching_ru = NumeralsMatching_ru()

    def test_make_agree_without_number_as_text(self):

        result = self.num_matching_ru.make_agree(1, ["деревянные", "рубли"])
        self.assertListEqual(["деревянный", "рубль"], result, "Некорректное склонение при единице")

        result = self.num_matching_ru.make_agree(2, ["деревянные", "рубли"])
        self.assertListEqual(["деревянных", "рубля"], result, "Некорректное склонение при двойке")

        result = self.num_matching_ru.make_agree(100, ["покрашенный", "золотой", "вагон"])
        self.assertListEqual(["покрашенных", "золотых", "вагонов"], result, "Некорректное склонение при сотни")

        result = self.num_matching_ru.make_agree(22, ["красивое", "красное", "пёрышко"])
        self.assertListEqual(["красивых", "красных", "пёрышка"], result, "Некорректное склонение при двадцати двух перышек")


    def test_make_agree_currency_and_measure(self):
        result = self.num_matching_ru.make_agree(1, ["килограммы"])
        self.assertListEqual(["килограмм"], result, "Некорректное склонение килограммов при единице")

        result = self.num_matching_ru.make_agree(-2, ["килограммы"])
        self.assertListEqual(["килограмма"], result, "Некорректное склонение килограммов при двойке")

        result = self.num_matching_ru.make_agree(2, ["гривна"])
        self.assertListEqual(["гривны"], result, "Некорректное склонение гривны при двойке")

        result = self.num_matching_ru.make_agree(100, ["гривна"])
        self.assertListEqual(["гривен"], result, "Некорректное склонение гривны при сотне")

        result = self.num_matching_ru.make_agree(-100, ["рубль"])
        self.assertListEqual(["рублей"], result, "Некорректное склонение рублей при сотне")

        result = self.num_matching_ru.make_agree(1, ["рублей"])
        self.assertListEqual(["рубль"], result, "Некорректное склонение рублей при единице")

        result = self.num_matching_ru.make_agree(2, ["рублей"])
        self.assertListEqual(["рубля"], result, "Некорректное склонение рублей при двойки")


    def test_make_agree_with_number_as_text(self):
        result = self.num_matching_ru.make_agree(1, ["добрый", "мальчугана"], number_as_text="одна")
        self.assertListEqual(["один", "добрый", "мальчуган"], result, "Некорректное склонение при единице")

        result = self.num_matching_ru.make_agree(2, ["добрый", "мальчуган"], number_as_text="две")
        self.assertListEqual(["два", "добрых", "мальчугана"], result, "Некорректное склонение при двойки")

        result = self.num_matching_ru.make_agree(-1, ["красивая", "девицы"], number_as_text="один")
        self.assertListEqual(["одна", "красивая", "девица"], result, "Некорректное склонение при единице")

        result = self.num_matching_ru.make_agree(-2, ["красивая", "девица"], number_as_text="два")
        self.assertListEqual(["две", "красивые", "девицы"], result, "Некорректное склонение при двойки")


    def test_make_agree_with_preposition(self):
        result = self.num_matching_ru.make_agree_with_preposition('с', 'одиннадцать ноль пять')
        self.assertEqual('одиннадцати ноль пяти', result, "Некорректное согласование с предлогом 'с'")

        result = self.num_matching_ru.make_agree_with_preposition('со', 'сто семьдесят два')
        self.assertEqual('сто семидесяти двух', result, "Некорректное согласование с предлогом 'со'")

        result = self.num_matching_ru.make_agree_with_preposition('до', 'двадцать два ноль ноль')
        self.assertEqual('двадцати двух ноль ноль', result, "Некорректное согласование с предлогом 'до'")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        del sys.argv[1:]

    unittest.main()
