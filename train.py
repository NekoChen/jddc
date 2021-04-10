from solver import *
from data_loader import get_loader
from configs import get_config, get_trans_config
from utils import Vocab
import os
import pickle


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    choose_model = 'mhred' # mhred
    if choose_model == 'trans':
        config = get_trans_config(mode='train')
        val_config = get_trans_config(mode='valid')
    else:
        config = get_config(mode='train')
        val_config = get_config(mode='valid')

    print(config)
    print(val_config)
    with open(os.path.join(config.save_path, 'config.txt'), 'w') as f:
        print(config, file=f)

    print('Loading Vocabulary...')
    vocab = Vocab()
    vocab.load(config.word2id_path, config.id2word_path)
    print(f'Vocabulary size: {vocab.vocab_size}')

    config.vocab_size = vocab.vocab_size

    train_data_loader = get_loader(
        sentences=load_pickle(config.sentences_path),
        images=load_pickle(config.images_path),
        conv_img_length=load_pickle(config.images_len_path),
        conversation_length=load_pickle(config.conversation_length_path),
        sentence_length=load_pickle(config.sentence_length_path),
        vocab=vocab,
        batch_size=config.batch_size,
        data_type='train'

    )

    eval_data_loader = get_loader(
        sentences=load_pickle(val_config.sentences_path),
        images=load_pickle(val_config.images_path),
        conv_img_length=load_pickle(val_config.images_len_path),
        conversation_length=load_pickle(val_config.conversation_length_path),
        sentence_length=load_pickle(val_config.sentence_length_path),
        vocab=vocab,
        batch_size=val_config.eval_batch_size,
        shuffle=False,
        data_type='eval'
    )

    solver = Solver

    solver = solver(config, train_data_loader, eval_data_loader, vocab=vocab, is_train=True)

    solver.build()
    solver.train()
