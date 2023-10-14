from fairseq.models.bart import BARTModel
bart = BARTModel.from_pretrained('/home/user/PhraseSimplification/bart.base', checkpoint_file='model.pt').cuda()
bart.eval()  # disable dropout (or leave in train mode to finetune)
print(bart.fill_mask(['Today is a good day, we <mask> to play,how about you?'], topk=3, beam=5))
