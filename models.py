import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import to_var, pad, normal_kl_div, normal_logpdf, bag_of_words_loss, to_bow, EOS_ID
import layers
import numpy as np
import random
def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask
class Transformer(nn.Module):
    def __init__(self,config):
        super(Transformer,self).__init__()
        self.config = config
        self.alpha = 0.7
        self.src_pad_idx, self.trg_pad_idx = self.config.pad_idx,self.config.pad_idx

        self.register_buffer('init_seq', torch.LongTensor([[self.config.sos_idx]]))
        self.register_buffer(
            'blank_seqs',
            torch.full((self.config.beam_size, self.config.max_unroll), self.config.pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.config.sos_idx
        self.register_buffer(
            'len_map',
            torch.arange(1, self.config.max_unroll + 1, dtype=torch.long).unsqueeze(0))

        trg_emb_prj_weight_sharing=config.proj_share_weight,
        emb_src_trg_weight_sharing=config.embs_share_weight,
        # 在“Attention Is All You Need”这篇文章的3.4节中，有这样的细节:
        # 在我们的模型中，我们在两者之间共享相同的权矩阵
        # 嵌入图层和pre-softmax线性变换…
        # 在嵌入层中，我们将这些权重乘以\sqrt{d_model}"。
        #
        # 选项:
        # 'emb':将\sqrt{d_model}嵌入输出
        # 'prj':乘(\sqrt{d_model} ^ -1)到线性投影输出
        # 'none':没有乘法

        assert config.scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (config.scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (config.scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = config.d_model

        self.encoder = layers.Trans_Encoder(
            n_src_vocab=config.vocab_size, n_position=config.n_position,
            d_word_vec=config.d_word_vec, d_model=config.d_model, d_inner=config.d_inner_hid,
            n_layers=config.n_layers, n_head=config.n_head, d_k=config.d_k, d_v=config.d_v,
            pad_idx=self.src_pad_idx, dropout=config.dropout, scale_emb=scale_emb)

        self.decoder = layers.Trans_Decoder(
            n_trg_vocab=config.vocab_size, n_position=config.n_position,
            d_word_vec=config.d_word_vec, d_model=config.d_model, d_inner=config.d_inner_hid,
            n_layers=config.n_layers, n_head=config.n_head, d_k=config.d_k, d_v=config.d_v,
            pad_idx=self.src_pad_idx, dropout=config.dropout, scale_emb=scale_emb)
        self.trg_word_prj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert config.d_model == config.d_word_vec, \
            'To facilitate the residual connections, \
             the dimensions of all module outputs shall be the same.'

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight

    def forward(self, input_sentences, input_sentence_length,
                input_conversation_length, target_sentences,
                input_images, input_images_length,
                decode=False):
        """
        Args:
            input_sentences: (Variable, LongTensor) [num_sentences, seq_len]
            target_sentences: (Variable, LongTensor) [num_sentences, seq_len]
        Return:
            decoder_outputs: (Variable, FloatTensor)
                - train: [batch_size, seq_len, vocab_size]
                - eval: [batch_size, seq_len]
        """
        src_mask = get_pad_mask(input_sentences, self.src_pad_idx)
        trg_mask = get_pad_mask(target_sentences, self.trg_pad_idx) & get_subsequent_mask(target_sentences)

        enc_output, *_ = self.encoder(input_sentences, src_mask)
        dec_output, *_ = self.decoder(target_sentences, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        # return seq_logit.view(-1, seq_logit.size(2))
        if not decode:
            return seq_logit

        else:
            prediction = self.generate(input_sentences)
            return prediction

    def _model_decode(self, trg_seq, enc_output, src_mask):
        trg_mask = get_subsequent_mask(trg_seq)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        return F.softmax(self.trg_word_prj(dec_output), dim=-1)

    def _get_init_state(self, src_seq, src_mask):
        beam_size = self.config.beam_size

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output = self._model_decode(self.init_seq, enc_output, src_mask)

        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)

        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        enc_output = enc_output.repeat(beam_size, 1, 1)
        return enc_output, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1

        beam_size = self.beam_size

        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)

        # Include the previous scores.
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)

        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores

    def generate(self, src_seq):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        batch_size = src_seq.size(0)

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.src_pad_idx
        max_seq_len, beam_size, alpha = self.config.max_unroll, self.config.beam_size, self.alpha

        with torch.no_grad():
            src_mask = get_pad_mask(src_seq, src_pad_idx)
            enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask)

            ans_idx = 0  # default
            for step in range(2, max_seq_len):  # decode up to max length
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask)
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)
                 # Check if all path finished
                 # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx
                 # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                    # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    # TODO: Try different terminate conditions.
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
        a =  gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()
        print(a)

        # for src_seq in input_sentences:
        #     src_pad_idx, trg_eos_idx = self.src_pad_idx, self.src_pad_idx
        #     max_seq_len, beam_size, alpha = self.config.max_unroll, self.config.beam_size, self.alpha
        #
        #     with torch.no_grad():
        #         src_mask = get_pad_mask(src_seq, src_pad_idx)
        #         enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask)
        #
        #         ans_idx = 0  # default
        #         for step in range(2, max_seq_len):  # decode up to max length
        #             dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask)
        #             gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)
        #
        #             # Check if all path finished
        #             # -- locate the eos in the generated sequences
        #             eos_locs = gen_seq == trg_eos_idx
        #             # -- replace the eos with its position for the length penalty use
        #             seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
        #             # -- check if all beams contain eos
        #             if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
        #                 # TODO: Try different terminate conditions.
        #                 _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
        #                 ans_idx = ans_idx.item()
        #                 break
        #     a =  gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()
        #     print(a)
        #


class MHRED(nn.Module):
    def __init__(self, config):
        super(MHRED, self).__init__()

        self.config = config
        self.encoder = layers.EncoderRNN(config.vocab_size,
                                         config.embedding_size,
                                         config.encoder_hidden_size,
                                         config.rnn,
                                         config.num_layers,
                                         config.bidirectional,
                                         config.dropout)

        context_input_size = (config.num_layers
                              * config.encoder_hidden_size
                              * self.encoder.num_directions)

        self.image_encoder = layers.ImageEncoder(context_input_size)

        self.context_encoder = layers.ContextRNN(context_input_size*2,
                                                 config.context_size,
                                                 config.rnn,
                                                 config.num_layers,
                                                 config.dropout)

        self.decoder = layers.DecoderRNN(config.vocab_size,
                                         config.embedding_size,
                                         config.decoder_hidden_size,
                                         config.rnncell,
                                         config.num_layers,
                                         config.dropout,
                                         config.word_drop,
                                         config.max_unroll,
                                         config.sample,
                                         config.temperature,
                                         config.beam_size)

        self.context2decoder = layers.FeedForward(config.context_size,
                                                  config.num_layers * config.decoder_hidden_size,
                                                  num_layers=1,
                                                  activation=config.activation)

        if config.tie_embedding:
            self.decoder.embedding = self.encoder.embedding

    def forward(self, input_sentences, input_sentence_length,
                input_conversation_length, target_sentences,
                input_images, input_images_length,
                decode=False):
        """
        Args:
            input_sentences: (Variable, LongTensor) [num_sentences, seq_len]
            target_sentences: (Variable, LongTensor) [num_sentences, seq_len]
        Return:
            decoder_outputs: (Variable, FloatTensor)
                - train: [batch_size, seq_len, vocab_size]
                - eval: [batch_size, seq_len]
        """
        num_sentences = input_sentences.size(0)
        max_len = input_conversation_length.data.max().item()

        batch_size = input_conversation_length.size(0)

        # encoder_outputs: [num_sentences, max_source_length, hidden_size * direction]
        # encoder_hidden: [num_layers * direction, num_sentences, hidden_size]
        encoder_outputs, encoder_hidden = self.encoder(input_sentences,
                                                       input_sentence_length)

        input_images = input_images.view(batch_size, -1, 3, 224, 224)
        input_images_length = input_images_length.view(batch_size, -1)

        indices = to_var(torch.tensor([i for i in range(max_len)]))

        input_images = input_images.index_select(1, indices)
        input_images = input_images.view(-1, 3, 224, 224)
        input_images_length = input_images_length.index_select(1, indices)

        img_encoder_outputs = self.image_encoder(input_images)
        img_encoder_outputs = img_encoder_outputs.view(batch_size, max_len, -1)


        # encoder_hidden: [num_sentences, num_layers * direction * hidden_size]
        encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(num_sentences, -1)

        # pad and pack encoder_hidden
        start = torch.cumsum(torch.cat((to_var(input_conversation_length.data.new(1).zero_()),
                                        input_conversation_length[:-1])), 0)

        # encoder_hidden: [batch_size, max_len, num_layers * direction * hidden_size]
        encoder_hidden = torch.stack([pad(encoder_hidden.narrow(0, s, l), max_len)
                                      for s, l in zip(start.data.tolist(),
                                                      input_conversation_length.data.tolist())], 0)

        comb_encoder_hidden = torch.cat([encoder_hidden, img_encoder_outputs], 2)

        # context_outputs: [batch_size, max_len, context_size]
        context_outputs, context_last_hidden = self.context_encoder(comb_encoder_hidden,
                                                                    input_conversation_length)

        # flatten outputs
        # context_outputs: [num_sentences, context_size]
        context_outputs = torch.cat([context_outputs[i, :l, :]
                                     for i, l in enumerate(input_conversation_length.data)])

        # project context_outputs to decoder init state
        decoder_init = self.context2decoder(context_outputs)

        # [num_layers, batch_size, hidden_size]
        decoder_init = decoder_init.view(self.decoder.num_layers, -1, self.decoder.hidden_size)

        # train: [batch_size, seq_len, vocab_size]
        # eval: [batch_size, seq_len]
        if not decode:

            decoder_outputs = self.decoder(target_sentences,
                                           init_h=decoder_init,
                                           decode=decode)
            return decoder_outputs

        else:
            # decoder_outputs = self.decoder(target_sentences,
            #                                init_h=decoder_init,
            #                                decode=decode)
            # return decoder_outputs.unsqueeze(1)
            # prediction: [batch_size, beam_size, max_unroll]
            prediction, final_score, length = self.decoder.beam_decode(init_h=decoder_init)

            # Get top prediction only
            # [batch_size, max_unroll]
            # prediction = prediction[:, 0]

            # [batch_size, beam_size, max_unroll]
            return prediction

    def generate(self, context, sentence_length, n_context, input_images, input_images_length):
        # context: [batch_size, n_context, seq_len]
        batch_size = context.size(0)
        # n_context = context.size(1)
        samples = []

        max_len = n_context

        input_images = input_images.view(batch_size, -1, 3, 224, 224)
        input_images_length = input_images_length.view(batch_size, -1)
        indices = to_var(torch.tensor([i for i in range(max_len)]))
        input_images = input_images.index_select(1, indices)
        input_images = input_images.view(-1, 3, 224, 224)
        input_images_length = input_images_length.index_select(1, indices)


        # Run for context
        context_hidden=None
        for i in range(n_context):
            # encoder_outputs: [batch_size, seq_len, hidden_size * direction]
            # encoder_hidden: [num_layers * direction, batch_size, hidden_size]
            encoder_outputs, encoder_hidden = self.encoder(context[:, i, :],
                                                           sentence_length[:, i])

            encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(batch_size, -1)

            input_image = input_images[i].unsqueeze(0)
            img_encoder_outputs = self.image_encoder(input_image)
            img_encoder_outputs = img_encoder_outputs.view(1, -1)

            comb_encoder_hidden = torch.cat([encoder_hidden, img_encoder_outputs], 1)

            # context_outputs: [batch_size, 1, context_hidden_size * direction]
            # context_hidden: [num_layers * direction, batch_size, context_hidden_size]
            context_outputs, context_hidden = self.context_encoder.step(comb_encoder_hidden,
                                                                        context_hidden)

        # Run for generation
        for j in range(self.config.n_sample_step):
            # context_outputs: [batch_size, context_hidden_size * direction]
            context_outputs = context_outputs.squeeze(1)
            decoder_init = self.context2decoder(context_outputs)
            decoder_init = decoder_init.view(self.decoder.num_layers, -1, self.decoder.hidden_size)

            prediction, final_score, length = self.decoder.beam_decode(init_h=decoder_init)
            # prediction: [batch_size, seq_len]
            prediction = prediction[:, 0, :]
            # length: [batch_size]
            length = [l[0] for l in length]
            length = to_var(torch.LongTensor(length))
            samples.append(prediction)

            encoder_outputs, encoder_hidden = self.encoder(prediction, length)
            encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(batch_size, -1)
            img_encoder_pad = torch.zeros_like(encoder_hidden)
            comb_encoder_hidden = torch.cat([encoder_hidden, img_encoder_pad], 1)            

            context_outputs, context_hidden = self.context_encoder.step(comb_encoder_hidden,
                                                                        context_hidden)

        samples = torch.stack(samples, 1)
        return samples
