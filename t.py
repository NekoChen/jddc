import os
import argparse
from pathlib import Path
project_dir = Path(__file__).resolve().parent
datasets_dir = project_dir.joinpath('./data/')
print(datasets_dir)
base_dir = os.path.dirname(os.path.realpath('__file__'))
print(base_dir)
print(project_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Maximum valid length of sentence
    # => SOS/EOS will surround sentence (EOS for source / SOS for target)
    # => maximum length of tensor = max_sentence_length + 1
    parser.add_argument('-s', '--max_sentence_length', type=int, default=50)
    parser.add_argument('-c', '--max_conversation_length', type=int, default=10)

    # Vocabulary
    parser.add_argument('--max_vocab_size', type=int, default=50000)
    parser.add_argument('--min_vocab_frequency', type=int, default=3)

    args = parser.parse_args()

    max_sent_len = args.max_sentence_length
    max_conv_len = args.max_conversation_length
    max_vocab_size = args.max_vocab_size
    min_freq = args.min_vocab_frequency

    print(max_sent_len)
    print(max_conv_len)
    print(max_vocab_size)
    print(min_freq)
