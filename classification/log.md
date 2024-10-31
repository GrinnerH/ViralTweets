(wwhenv38) robot@Ren:~/wwh/ViralTweets/classification/model_with_extra_features$ python3 classification.py 
INFO:root:Number of viral tweets tweeted on the same day 787
INFO:root:Number of non viral tweets tweeted on the same day 787
INFO:root:1,259 training samples with   630 positives and   629 negatives
INFO:root:  314 validation samples with   157 positives and   157 negatives
Generating train split: 1259 examples [00:00, 283858.99 examples/s]
Generating test split: 314 examples [00:00, 96307.97 examples/s]
/home/robot/anaconda3/envs/wwhenv38/lib/python3.8/site-packages/huggingface_hub/file_download.py:1150: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
pytorch_model.bin: 100%|█████████████████████████████████████████████████████████████████████████| 543M/543M [00:54<00:00, 10.0MB/s]
Some weights of the model checkpoint at vinai/bertweet-base were not used when initializing RobertaModel: ['lm_head.bias', 'lm_head.decoder.bias', 'lm_head.layer_norm.weight', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight', 'lm_head.dense.weight', 'lm_head.dense.bias']
- This IS expected if you are initializing RobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Map: 100%|████████████████████████████████████████████████████████████████████████████| 1259/1259 [00:00<00:00, 10902.30 examples/s]
Map: 100%|███████████████████████████████████████████████████████████████████████████████| 314/314 [00:00<00:00, 8997.09 examples/s]
Downloading builder script: 4.20kB [00:00, 8.51MB/s]                                                                                
Downloading builder script: 7.36kB [00:00, 16.3MB/s]                                                    | 0.00/1.67k [00:00<?, ?B/s]
Downloading builder script: 7.55kB [00:00, 12.2MB/s]
Downloading builder script: 6.77kB [00:00, 15.9MB/s]
/home/robot/anaconda3/envs/wwhenv38/lib/python3.8/site-packages/huggingface_hub/file_download.py:1150: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
config.json: 100%|█████████████████████████████████████████████████████████████████████████████████| 285/285 [00:00<00:00, 36.7kB/s]
vocab.txt: 232kB [00:00, 361kB/s]                                                                         | 0.00/285 [00:00<?, ?B/s]
pytorch_model.bin: 100%|███████████████████████████████████████████████████████████████████████| 17.8M/17.8M [00:01<00:00, 10.3MB/s]
Some weights of the model checkpoint at prajjwal1/bert-tiny were not used when initializing BertModel: ['cls.seq_relationship.bias', 'cls.predictions.decoder.weight', 'cls.predictions.transform.LayerNorm.weight', 'cls.seq_relationship.weight', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.dense.bias', 'cls.predictions.bias', 'cls.predictions.decoder.bias']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
                                                                                                                                   Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Map: 100%|████████████████████████████████████████████████████████████████████████████| 1259/1259 [00:00<00:00, 58526.69 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████| 314/314 [00:00<00:00, 27959.65 examples/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [00:48<00:00, 12.36it/s]
You're using a BertTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
                                                                                                                                   /home/robot/anaconda3/envs/wwhenv38/lib/python3.8/site-packages/huggingface_hub/file_download.py:1150: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
tokenizer_config.json: 100%|█████████████████████████████████████████████████████████████████████| 49.0/49.0 [00:00<00:00, 33.3kB/s]
config.json: 570B [00:00, 418kB/s]                                                                                                  
vocab.txt: 213kB [00:00, 353kB/s] 
tokenizer.json: 436kB [00:00, 475kB/s] 
pytorch_model.bin: 100%|█████████████████████████████████████████████████████████████████████████| 436M/436M [00:52<00:00, 8.29MB/s]
Some weights of the model checkpoint at bert-base-cased were not used when initializing BertModel: ['cls.seq_relationship.bias', 'cls.predictions.decoder.weight', 'cls.predictions.transform.LayerNorm.weight', 'cls.seq_relationship.weight', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.dense.bias', 'cls.predictions.bias']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Map: 100%|████████████████████████████████████████████████████████████████████████████| 1259/1259 [00:00<00:00, 59869.04 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████| 314/314 [00:00<00:00, 31052.07 examples/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [01:11<00:00,  8.40it/s]
You're using a BertTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
100%|████████████████████████████████████████████████████████████████████████████████████████████▊| 599/600 [00:34<00:00, 17.03it/s]/home/robot/anaconda3/envs/wwhenv38/lib/python3.8/site-packages/huggingface_hub/file_download.py:1150: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
tokenizer_config.json: 100%|█████████████████████████████████████████████████████████████████████| 25.0/25.0 [00:00<00:00, 17.9kB/s]
config.json: 100%|██████████████████████████████████████████████████████████████████████████████████| 481/481 [00:00<00:00, 358kB/s]
vocab.json: 899kB [00:01, 492kB/s]                                                                        | 0.00/481 [00:00<?, ?B/s]
merges.txt: 456kB [00:00, 568kB/s] 
tokenizer.json: 1.36MB [00:02, 651kB/s]███████████████████████████████████████████████████████████| 600/600 [00:49<00:00, 17.03it/s]
pytorch_model.bin: 100%|█████████████████████████████████████████████████████████████████████████| 501M/501M [00:49<00:00, 10.0MB/s]
Some weights of the model checkpoint at roberta-base were not used when initializing RobertaModel: ['lm_head.bias', 'lm_head.layer_norm.weight', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight', 'lm_head.dense.weight', 'lm_head.dense.bias']
- This IS expected if you are initializing RobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Map: 100%|████████████████████████████████████████████████████████████████████████████| 1259/1259 [00:00<00:00, 71155.04 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████| 314/314 [00:00<00:00, 50697.18 examples/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [01:42<00:00,  5.85it/s]
You're using a RobertaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
                                                                                                                                   INFO:root:checkpoint: vinai/bertweet-base: {'accuracy': 0.7388535031847133, 'recall': 0.8407643312101911, 'precision': 0.6984126984126984, 'f1': 0.7630057803468209}
checkpoint: prajjwal1/bert-tiny: {'accuracy': 0.7229299363057324, 'recall': 0.8662420382165605, 'precision': 0.6732673267326733, 'f1': 0.7576601671309192}
checkpoint: bert-base-cased: {'accuracy': 0.7101910828025477, 'recall': 0.802547770700637, 'precision': 0.6774193548387096, 'f1': 0.7346938775510203}
checkpoint: roberta-base: {'accuracy': 0.7261146496815286, 'recall': 0.802547770700637, 'precision': 0.6961325966850829, 'f1': 0.7455621301775148}

100%|█████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [00:40<00:00, 14.70it/s]
