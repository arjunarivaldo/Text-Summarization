# model_architecture.py
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss

class BertSumClassifier(BertPreTrainedModel):
    """
    Model BERT Kustom untuk Klasifikasi Kalimat (Extractive Summarization).
    
    Model ini menambahkan 'Sentence Position Embeddings' ke 'Token Embeddings'
    standar BERT sebelum memasukkannya ke transformer.
    (Berdasarkan EDA Langkah 2: 'Lead Bias')
    """
    def __init__(self, config):
        super().__init__(config)
        
        self.bert = BertModel(config, add_pooling_layer=False)
        
        # LAYER BARU: Sentence Position Embeddings (EDA 2)
        # Asumsi maks 128 kalimat per artikel
        self.sent_pos_embeddings = nn.Embedding(128, config.hidden_size)
        
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        sentence_pos_ids=None, # <--- Input kustom kita
        labels=None,
        **kwargs
    ):
        
        word_embeddings = self.bert.embeddings.word_embeddings(input_ids)
        
        # Pastikan sentence_pos_ids tidak melebihi batas embedding (128)
        sent_pos_ids = torch.clamp(sentence_pos_ids, 0, 127)
        sent_pos_embeds = self.sent_pos_embeddings(sent_pos_ids)
        
        # GABUNGKAN KEDUA EMBEDDING
        combined_embeddings = word_embeddings + sent_pos_embeds
        
        outputs = self.bert(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            # Kita gunakan 'ignore_index' -100 untuk token non-[CLS]
            loss_fct = CrossEntropyLoss(ignore_index=-100) 
            flat_logits = logits.view(-1, self.config.num_labels)
            flat_labels = labels.view(-1)
            loss = loss_fct(flat_logits, flat_labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )