import time
from text_preparation.framing_numbers.tests.fixtures import test_cases, test_texts
from text_preparation.framing_numbers.framing_numbers import NumberWithSpacesCase


def benchmark():
    framer = NumberWithSpacesCase()
    start_time = time.time()
    N = 1000
    for i in range(0, N):
        for text in test_texts.keys():
            _ = framer.frame_numbers(text)
    elapsed_time = time.time() - start_time
    queries_count = N * len(test_texts)
    per_query = elapsed_time / queries_count
    print("Total: {:.2f} msec for {} queries\nMean time per query: {:.6f} msec".format(elapsed_time * 1000, queries_count, per_query * 1000))


if __name__ == '__main__':
    benchmark()
