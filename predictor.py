import torch
from torch import nn
from utils.checkpointing import load_checkpoint, load_vocab
from models.model import SummarizationModel
from tqdm import tqdm
from data.dataset import *
from beam_search import BeamSearch


class Predictor(object):
    def __init__(self, hparams, model=None, vocabs=None, checkpoint=None):
        super(Predictor, self).__init__()
        self.hparams = hparams
        self.model = model
        self.vocab_word = vocabs
        self.device = hparams.device
        self.batch_size = hparams.batch_size

        # Beam-search configuration
        self.min_length = hparams.min_length
        self.max_length = hparams.max_length
        self.beam_size = hparams.beam_size
        self.start_token_id = self.vocab_word.token2id['<BEGIN>']
        self.end_token_id = self.vocab_word.token2id['<END>']

        self.device = hparams.device

        if (model == None) and (checkpoint is not None):
            self.build_model()
            if self.vocab_word is None:
                self.vocab_word = load_vocab(self.hparams.vocab_word_path)
            model_state_dict, optimizer_state_dict = load_checkpoint(self.hparams.load_pthpath)
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(model_state_dict)
            else:
                self.model.load_state_dict(model_state_dict)

        self.model.eval()

    def build_model(self):
        # Define model
        self.model = SummarizationModel(self.hparams, self.vocab_word)
        self.embedding_word = self.model.embedding_word

        # Multi-GPU
        self.model = self.model.to(self.device)

        # # Use Multi-GPUs
        if -1 not in self.hparams.gpu_ids and len(self.hparams.gpu_ids) > 1:
            self.model = nn.DataPzarallel(self.model, self.hparams.gpu_ids)

    def generator(self, decoder_outputs):
        # Reuse the weight of embedding matrix D, to decode v_{k-1} into a probability distribution
        logits = torch.matmul(decoder_outputs, torch.transpose(self.model.embedding_word.weight, 0, 1))
        softmax = nn.LogSoftmax(dim=-1)
        probs = softmax(logits)
        return logits, probs

    def get_summaries(self, idxs):
        tokens = [self.vocab_word.id2token[idx.item()] for idx in idxs]
        summary = ' '.join(tokens)
        return summary

    def inference(self, inputs, src_masks):

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (self.beam_size - 1),
                         device=self.device).repeat(self.batch_size))
        alive_seq = torch.full(
            [self.batch_size * self.beam_size, 1],
            self.start_token_id,
            dtype=torch.long,
            device=self.device)
        batch_offset = torch.arange(
            self.batch_size, dtype=torch.long, device=self.device)
        beam_offset = torch.arange(
            0,
            self.batch_size * self.beam_size,
            step=self.beam_size,
            dtype=torch.long,
            device=self.device)

        hypotheses = [[] for _ in range(self.batch_size)]
        results = {}
        results["predictions"] = [[] for _ in range(self.batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(self.batch_size)]  # noqa: F812
        results["gold_score"] = [0] * self.batch_size

        inputs = torch.squeeze(inputs, 0)  # [num_turns, seq_len]
        inputs_word_emb = self.model.embedding_word(inputs)

        src_masks = src_masks.squeeze(0)
        word_level_outputs = self.model.word_level_encoder(inputs=inputs_word_emb,
                                                           src_masks=src_masks)  # [num_turns, seq_len, 300]

        turn_level_inputs = word_level_outputs[:, 0]  # [num_turns, 300]
        turn_level_inputs = torch.unsqueeze(turn_level_inputs, 0)  # [1, num_turns, 300]
        turn_level_outputs = self.model.turn_level_encoder(turn_level_inputs)  # [1, num_turns, 300]

        word_level_shape = word_level_outputs.shape
        word_level_outputs = word_level_outputs.view(word_level_shape[0] * word_level_shape[1], 300)
        word_level_outputs = word_level_outputs.unsqueeze(0)  # [1, num_turns * seq_len, 300]

        decoder_state = self.model.decoder.init_decoder_state()

        word_level_memory_beam = word_level_outputs.detach().repeat(self.beam_size, 1, 1)  # [beam_size, num_turns * seq_len, 300]
        turn_level_memory_beam = turn_level_outputs.detach().repeat(self.beam_size, 1, 1)  # [beam_size, num_turns, 300]

        for step in range(self.max_length):
            print('[Step]: ', step)
            print('alive_seq[:, -1].view(1, -1).transpose(0, 1): ', alive_seq[:, -1].view(1, -1).transpose(0, 1))
            tgt_inputs = alive_seq[:, -1].view(1, -1).transpose(0, 1)  # (beam_size, tgt_seq_len==1)
            tgt_word_emb = self.model.embedding_word(tgt_inputs)

            decoder_outputs, decoder_state = self.model.decoder(
                inputs=(tgt_word_emb, word_level_memory_beam, turn_level_memory_beam),
                state=decoder_state)

            logits, log_probs = self.generator(decoder_outputs)  # logits: [beam_size, tgt_seq_len==1, vocab_size]
            log_probs = log_probs.squeeze(1) # [beam_size, vocab_size]
            vocab_size = log_probs.size(1)

            max_token = torch.argmax(log_probs[0])
            print('arg_max: ', max_token, 'Max_token: ', self.vocab_word.id2token[max_token.item()])

            if step < self.min_length:
                log_probs[:, self.end_token_id] = -1e20

            # print('log_probs.shape: ', log_probs.shape)
            # print('topk_log_probs.view(-1).unsqueeze(1) shape: ', topk_log_probs.view(-1).unsqueeze(1).shape)

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = 0.6
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            if self.hparams.blook_trigram:
                # Trigram-Blocking
                cur_len = alive_seq.size(1)
                if (cur_len > 3):
                    for i in range(alive_seq.size(0)):  # For each (batch x beam_size)
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        words = [self.vocab_word.id2token[w] for w in words]
                        words = ' '.join(words).split()
                        if (len(words) <= 3):
                            continue
                        trigrams = [(words[i - 1], words[i], words[i + 1]) for i in range(1, len(words) - 1)]
                        trigram = tuple(trigrams[-1])
                        # print('step:', step)
                        # print('i: ', i)
                        # print('trigrams[:-1]: ', trigrams[:-1])
                        # print('trigram: ', trigram)
                        # print('\n')
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            curr_scores[i] = -10e20


            curr_scores = curr_scores.reshape(-1, self.beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(self.beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            print('topk_ids: ', topk_ids)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)

            is_finished = topk_ids.eq(self.end_token_id)
            if step + 1 == self.max_length:
                is_finished.fill_(1)
            end_condition = is_finished[:, 0].eq(1)

            if is_finished.any():
                predictions = alive_seq.view(-1, self.beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]

                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)

                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break

                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))

            # Reorder states.
            select_indices = batch_index.view(-1)
            word_level_memory_beam = word_level_memory_beam.index_select(0, select_indices)
            turn_level_memory_beam = turn_level_memory_beam.index_select(0, select_indices)

            decoder_state.beam_update(select_indices)

        preds = results['predictions'][0][0]
        summary = self.get_summaries(preds)
        print(summary)

    # def inference(self, inputs, src_masks):
    #     inputs = torch.squeeze(inputs, 0)  # [num_turns, seq_len]
    #
    #     # print('inputs shape: ', inputs.shape)
    #
    #     inputs_word_emb = self.model.embedding_word(inputs)
    #
    #     src_masks = src_masks.squeeze(0)
    #     word_level_outputs = self.model.word_level_encoder(inputs=inputs_word_emb,
    #                                                  src_masks=src_masks)  # [num_turns, seq_len, 300]
    #
    #     turn_level_inputs = word_level_outputs[:, 0]  # [num_turns, 300]
    #     turn_level_inputs = torch.unsqueeze(turn_level_inputs, 0)  # [1, num_turns, 300]
    #     turn_level_outputs = self.model.turn_level_encoder(turn_level_inputs)  # [1, num_turns, 300]
    #
    #     word_level_shape = word_level_outputs.shape
    #     word_level_outputs = word_level_outputs.view(word_level_shape[0] * word_level_shape[1], 300)
    #     word_level_outputs = word_level_outputs.unsqueeze(0) # [1, num_turns * seq_len, 300]
    #
    #     decoder_state = self.model.decoder.init_decoder_state()
    #
    #     word_level_memory_beam = word_level_outputs.detach().repeat(self.beam_size, 1, 1) # [beam_size, num_turns * seq_len, 300]
    #     turn_level_memory_beam = turn_level_outputs.detach().repeat(self.beam_size, 1, 1) # [beam_size, num_turns, 300]
    #
    #     beam = BeamSearch(self.hparams, self.vocab_word, n_top=self.hparams.n_top)
    #
    #     for _ in range(self.max_length):
    #         tgt_inputs = beam.get_current_state().unsqueeze(1) # (beam_size, tgt_seq_len==1)
    #         tgt_inputs = tgt_inputs.to(self.device)
    #         tgt_word_emb = self.model.embedding_word(tgt_inputs)
    #         decoder_outputs, decoder_state = self.model.decoder(inputs=(tgt_word_emb, word_level_memory_beam, turn_level_memory_beam),
    #                                                             state=decoder_state)
    #
    #         logits, probs = self.generator(decoder_outputs) # logits: [beam_size, 1, vocab_size]
    #         attention = self.model.decoder.decoder_layers[-1].multi_head_attention_turn.attention # [beam_size, num_heads, tgt_seq_len==1, num_turns]
    #
    #         # remove dimension of num_heads and reshape to (tgt_seq_len, beam_size, num_turns)
    #
    #         beam.advance(logits.squeeze(1), attention)
    #         # beam.advance(probs.squeeze(1), attention)
    #
    #         beam_current_origin = beam.get_current_origin()  # (beam_size, )
    #         decoder_state.beam_update(beam_current_origin)
    #
    #         if beam.is_finish():
    #             break
    #
    #     scores, ks = beam.sort_finished(minimum=self.hparams.n_top)
    #     hypothesises = []
    #     for i, (times, k) in enumerate(ks[:self.hparams.n_top]):
    #         hypothesis, attention = beam.get_hypothesis(times, k)
    #         hypothesises.append(hypothesis)
    #
    #     print('hypothesises: ', hypothesises[0])
    #
    #     # hypothesises = [[token.item() for token in h] for h in hypothesises]
    #     summaries = [self.get_summaries(hypothesis) for hypothesis in hypothesises]
    #     summaries = list(reversed(summaries))
    #     # print("summaries: ", summaries)


    # def evaluate(self, test_dataloader):
    #     with torch.no_grad():
    #         for batch_idx, batch in enumerate(tqdm(test_dataloader)):
    #             data = batch
    #             dialogues_ids = data['dialogues_ids'].to(self.device)
    #             labels_ids = data['labels_ids'].to(self.device)  # [batch, tgt_seq_len]
    #             src_masks = data['src_masks'].to(self.device)
    #
    #             # Beam-search starts with <BEGIN> token
    #
    #
    #             # Block trigram


