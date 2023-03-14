import torchvision.models as models
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


# class BigModel(nn.Module):
#     def __init__(self, main_model):
#         super(BigModel, self).__init__()
#         self.main_model = main_model
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, tok, att, cud=True):
#         typ = torch.zeros(tok.shape).long()
#         if cud:
#             typ = typ.cuda()
#         pooled_output = self.main_model(tok, token_type_ids=typ, attention_mask=att)['pooler_output']
#         logits = self.dropout(pooled_output)
#         return logits


# class TextEncoder(nn.Module):
#     def __init__(self, pretrained=True):
#         super(TextEncoder, self).__init__()
#         if pretrained:  # if use pretrained scibert model
#             self.main_model = BertModel.from_pretrained('bert_pretrained/')
#         else:
#             config = BertConfig(vocab_size=31090, )
#             self.main_model = BertModel(config)

#         self.dropout = nn.Dropout(0.1)
#         # self.hidden_size = self.main_model.config.hidden_size

#     def forward(self, input_ids, attention_mask):
#         device = input_ids.device
#         typ = torch.zeros(input_ids.shape).long().to(device)
#         output = self.main_model(input_ids, token_type_ids=typ, attention_mask=attention_mask)['pooler_output']  # b,d
#         logits = self.dropout(output)
#         return logits 

from transformers import DistilBertModel, DistilBertTokenizer


class TextEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(TextEncoder, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("nlpie/distil-biobert")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 4)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


if __name__ == '__main__':
    model = TextEncoder()