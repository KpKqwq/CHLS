from transformers import BertTokenizer, BartForConditionalGeneration
from torch import nn
import numpy as np
import torch
tokenizer = BertTokenizer.from_pretrained("bart-base-chinese")
class ParaBART:
    def __init__(self, device='cuda:0', min_length=2, max_length=256, checkpoint='bart-base-chinese'):
        # Set up model
        self.device = device
        self.min_length = min_length
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def generation(self, original_sent, prefix, beam_size):
        input_ids = self.tokenizer(original_sent, return_tensors="pt")["input_ids"].to(self.device)
        prefix_ids = self.tokenizer(prefix, return_tensors="pt")["input_ids"][0][:-1].to(self.device)

        outputs, scores = self.model.generate(
            input_ids=input_ids,
            prefix_ids=prefix_ids,
            num_beams=beam_size,
            min_length=self.min_length,
            max_length=self.max_length,
            early_stopping=True
        )

        prefix_paraphrases = [
            self.tokenizer.batch_decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            for output in outputs
        ]
        return prefix_paraphrases, scores

    def generation_atten(self, original_sent, prefix, beam_size):
        input_ids = self.tokenizer(original_sent, return_tensors="pt")["input_ids"].to(self.device)
        prefix_ids = self.tokenizer(prefix, return_tensors="pt")["input_ids"][0][:-1].to(self.device)

        outputs, scores = self.model.generate(
            input_ids=input_ids,
            prefix_ids=prefix_ids,
            num_beams=beam_size,
            min_length=self.min_length,
            max_length=self.max_length,
            early_stopping=True
        )

        prefix_paraphrases = [
            self.tokenizer.batch_decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            for output in outputs
        ]

        return prefix_paraphrases, scores

    def generation_original(self, original_sent, beam_size):
        input_ids = self.tokenizer(original_sent, return_tensors="pt").to(self.device)["input_ids"].to(self.device)

        output = self.model.generate(
            input_ids=input_ids,
            num_beams=beam_size,
            min_length=self.min_length,
            max_length=self.max_length,
            num_return_sequences=beam_size
        )

        original_paraphrases = self.tokenizer.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return original_paraphrases

    def score(self, srcs, tgts, batch_size=4):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=256,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=256,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list
generateion_tool=ParaBART()
source_word="截然"
word="完全"
ori_text="截然"
ori_text.replace(" ","")
print(np.exp(generateion_tool.score([ori_text],[ori_text.replace(source_word,word)], batch_size=1)[0]))