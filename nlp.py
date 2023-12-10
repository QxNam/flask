import joblib
import torch
import torch.nn as nn
from torcheval.metrics.functional import multiclass_f1_score
import transformers
from underthesea import word_tokenize
from huggingface_hub import login
bert_pretrain = 'vinai/phobert-base'
model_predict = r'./model/nlp/model.bin'
meta = r'./model/nlp/meta.bin'
maxlen = 128
key = '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'
TOKENIZER = transformers.AutoTokenizer.from_pretrained(
    bert_pretrain,
    do_lower_case=True
)

login(key, add_to_git_credential=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class EntityDataset:
    def __init__(self, texts, ner, nested):
        self.texts = texts
        self.ner = ner
        self.nested = nested
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = self.texts[item]
        ner = self.ner[item]
        nested = self.nested[item]

        ids = []
        target_ner = []
        target_nested =[]

        for i, s in enumerate(text):
            inputs = TOKENIZER.encode(
                s,
                add_special_tokens=False
            )
            # abhishek: ab ##hi ##sh ##ek
            input_len = len(inputs)
            ids.extend(inputs)
            target_ner.extend([ner[i]] * input_len)
            target_nested.extend([nested[i]] * input_len)

        ids = ids[:maxlen - 2]
        target_ner = target_ner[:maxlen - 2]
        target_nested = target_nested[:maxlen - 2]

        ids = [0] + ids + [2]
        target_ner = [1] + target_ner + [1]
        target_nested = [1] + target_nested + [1]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = maxlen - len(ids)

        ids = ids + ([1] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_ner = target_ner + ([1] * padding_len)
        target_nested = target_nested + ([1] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_ner": torch.tensor(target_ner, dtype=torch.long),
            "target_nested": torch.tensor(target_nested, dtype=torch.long),
        }

def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss(ignore_index=1)
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)  
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(1).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss

def accuracy_fn(output, target, mask, num_labels):
    _, predicted = torch.max(output, 2)
    correct = (predicted == target) & (target != 1) & (mask == 1)
    num_correct = correct.sum().item()
    num_samples = target[(target != 1) & (mask == 1)].size()[0]
    accuracy = num_correct / num_samples if num_samples > 0 else 0.0
    return accuracy

def f1score_fn(output, target, mask, num_labels):
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)  
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(1).type_as(target)
    )
    f1 = multiclass_f1_score(active_logits, active_labels, num_classes=num_labels, average="macro")
    return f1

class EntityModel(nn.Module):
    def __init__(self, num_nested, num_ner):
        super(EntityModel, self).__init__()
        self.num_nested = num_nested
        self.num_ner = num_ner
        self.bert = transformers.AutoModel.from_pretrained(bert_pretrain)

        self.bert_drop_1 = nn.Dropout(0.3)
        self.bert_drop_2 = nn.Dropout(0.3)
        self.out_nested = nn.Linear(768, self.num_nested)
        self.out_ner = nn.Linear(768, self.num_ner)
    
    def forward(self, ids, mask, token_type_ids, target_ner, target_nested):
        outputs = self.bert(
            ids, 
            attention_mask=mask, 
            token_type_ids=token_type_ids
        )
        o1 = outputs["last_hidden_state"]
        bo_nested = self.bert_drop_1(o1)
        bo_ner = self.bert_drop_2(o1)

        nested = self.out_nested(bo_nested)
        ner = self.out_ner(bo_ner)

        loss_nested = loss_fn(nested, target_nested, mask, self.num_nested)
        loss_ner = loss_fn(ner, target_ner, mask, self.num_ner)

        loss = (loss_nested + loss_ner) / 2
        
        acc_nested = accuracy_fn(nested, target_nested, mask, self.num_nested)
        acc_ner = accuracy_fn(ner, target_ner, mask, self.num_ner)

        acc = (acc_nested + acc_ner) / 2
        
        f1_nested = f1score_fn(nested, target_nested, mask, self.num_nested)
        f1_ner = f1score_fn(ner, target_ner, mask, self.num_ner)

        f1_score = (f1_nested + f1_ner) / 2

        return nested, ner, loss, acc, f1_score

meta_data = joblib.load(meta)
enc_ner = meta_data["enc_ner"]
enc_nested = meta_data["enc_nested"]

num_ner = len(list(enc_ner.classes_))
num_nested = len(list(enc_nested.classes_))



def predict(sentence):
    sentence = word_tokenize(sentence, format="text")
    tokenized_sentence = TOKENIZER.encode(sentence)

    sentence = sentence.split()
    test_dataset = EntityDataset(
        texts=[sentence], 
        ner=[[1] * len(sentence)], 
        nested=[[1] * len(sentence)]
    )
    model = EntityModel(num_nested=num_nested, num_ner=num_ner)
    model.load_state_dict(torch.load(model_predict, map_location=torch.device('cpu')))
    model.to(device)
    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        nested, ner, _, _, _ = model(**data)
        predict_nested = enc_nested.inverse_transform(
                                nested.argmax(2).cpu().numpy().reshape(-1))[:len(tokenized_sentence)]
        predict_ner = enc_ner.inverse_transform(
                            ner.argmax(2).cpu().numpy().reshape(-1))[:len(tokenized_sentence)]
    res = []
    for word, ner_main, ner_nested in zip(sentence, predict_ner[1:-1], predict_nested[1:-1]):
        w = f'{word} '
        check = False
        if ner_main!='O':
            check = True
            w += f'[{ner_main}'
            if ner_nested!='O':
                w += f', {ner_nested}] '
            else:
                w += '] '
        res.append((w.replace('_', ' '), check))
    return res

# <script>
#         $('#nlpForm').on('submit', function(e) {
#             e.preventDefault();
#             $.ajax({
#                 url: '/nlp_predict',
#                 type: 'POST',
#                 data: $(this).serialize(),
#                 success: function(response) {
#                     var htmlContent = $(response.text);
#                     $('.output').html(response.text);
#                 },
#                 error: function(error) {
#                     console.log(error);
#                 }
#             });
#         });
#     </script>