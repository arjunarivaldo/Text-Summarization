# src/model_abstractive_kustom.py
# Arsitektur T5 Kustom dengan Sentence Position Embeddings

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import Optional, Tuple, Union

class T5WithSentencePosition(T5ForConditionalGeneration):
    """
    Model T5 Kustom yang dimodifikasi untuk menerima Sentence Position Embeddings.
    Logika ini meniru arsitektur BertSumClassifier.
    """
    def __init__(self, config):
        super().__init__(config)

        # --- KUSTOMISASI KITA (dari EDA 2) ---
        # Menambahkan layer embedding posisi kalimat
        # d_model adalah nama T5 untuk 'hidden_size'
        self.max_sent_positions = 128 
        self.sent_pos_embeddings = nn.Embedding(self.max_sent_positions, config.d_model)
        # -------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,

        # --- INPUT KUSTOM KITA ---
        sentence_pos_ids: Optional[torch.LongTensor] = None, 
        # -------------------------

        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # --- KUSTOMISASI INTI (HANYA JIKA ENCODER BERJALAN) ---
        if encoder_outputs is None:
            # 1. Dapatkan embedding kata standar dari T5
            if inputs_embeds is None:
                inputs_embeds = self.encoder.embed_tokens(input_ids)

            # 2. Dapatkan embedding posisi kalimat kustom
            if sentence_pos_ids is None:
                # Jika tidak ada pos_ids (saat .generate()), buat tensor nol
                sent_pos_embeds = torch.zeros_like(inputs_embeds)
            else:
                sent_pos_ids_clamped = torch.clamp(sentence_pos_ids, max=self.max_sent_positions - 1)
                sent_pos_embeds = self.sent_pos_embeddings(sent_pos_ids_clamped)

            # 3. Gabungkan embedding
            combined_embeddings = inputs_embeds + sent_pos_embeds

            # 4. Masukkan embedding GABUNGAN ke ENCODER
            encoder_outputs = self.encoder(
                inputs_embeds=combined_embeddings, # <<< KEY CHANGE
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # --- AKHIR KUSTOMISASI ---

        # Sisa dari fungsi ini adalah standar T5 (bagian Decoder)
        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        # Bagian Decoder 
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        # Perhitungan Loss
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )