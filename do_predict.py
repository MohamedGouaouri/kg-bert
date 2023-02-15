# contains functions that call the kgbert model to make predictions. 
# This requires the linkedin_output folder to contain the result of training :
# - config.json
# - pytorch_model.bin
# - vocab.txt

import os
import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from tqdm import tqdm
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from KGProcessor import KGProcessor, InputExample, InputFeatures, _truncate_seq_triple

# where entities and descriptions are stored
DATA_DIR = './data/linkedin/'
ENTITIES_FILE = './data/linkedin/entities.txt'
# where the model is stored
OUTPUT_DIR = './linkedin_output/'
# KGProcessor has the util functions
kg_processor = KGProcessor()
label_list = kg_processor.get_labels(DATA_DIR)
num_labels = len(label_list)
max_seq_length = 128

tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIR, do_lower_case=True)
model = model = BertForSequenceClassification.from_pretrained(OUTPUT_DIR, num_labels=num_labels)


def _create_example(input1, input2, data_dir, guid):
    """Creates example for evaluation pair."""
    # entity to text
    ent2text = {}
    with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
        ent_lines = f.readlines()
        for line in ent_lines:
            temp = line.strip().split('\t')
            if len(temp) == 2:
                ent2text[temp[0]] = temp[1]

    entities = list(ent2text.keys())

    rel2text = {}
    with open(os.path.join(data_dir, "relation2text.txt"), 'r') as f:
        rel_lines = f.readlines()
        for line in rel_lines:
            temp = line.strip().split('\t')
            rel2text[temp[0]] = temp[1]
            ### Change the following line if the eval_data has more than 2 columns
            relation_text = temp[1]

    
    try :
        head_ent_text = ent2text[input1]
    except KeyError :
        head_ent_text = input1

    try :
        tail_ent_text = ent2text[input2]
    except KeyError :
        tail_ent_text = input2
    # relation_text = rel2text[line[1]]

    # This is for the case where eval_data has two columns, and entities that don't have descriptions
    # head_ent_text = line[0]
    # tail_ent_text = line[1]

    # guid = "%s-%s" % ("eval", i)
    text_a = head_ent_text
    text_b = relation_text
    text_c = tail_ent_text
    
    return InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=None)

def convert_example_to_feature(example, label_list, max_seq_length, tokenizer, print_info=True):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    tokens_c = None

    if example.text_b and example.text_c:
        tokens_b = tokenizer.tokenize(example.text_b)
        tokens_c = tokenizer.tokenize(example.text_c)
        _truncate_seq_triple(tokens_a, tokens_b,
                                tokens_c, max_seq_length - 4)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)
    if tokens_c:
        tokens += tokens_c + ["[SEP]"]
        segment_ids += [0] * (len(tokens_c) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)


def predict_kgbert_singlepair(input1, input2, eval_id) :

    eval_example = _create_example(input1, input2, DATA_DIR, guid="%s-%s" % ("eval", eval_id))
    eval_feature = convert_example_to_feature(eval_example, label_list, max_seq_length, tokenizer)

    input_id = torch.tensor([eval_feature.input_ids], dtype=torch.long)
    input_mask = torch.tensor([eval_feature.input_mask], dtype=torch.long)
    segment_id = torch.tensor([eval_feature.segment_ids], dtype=torch.long)

    device = torch.device('cpu')
    model.to(device)
    model.eval()

    input_id = input_id.to(device)
    input_mask = input_mask.to(device)
    segment_id = segment_id.to(device)

    preds = []
    probs = []

    with torch.no_grad():
        logits, *_ = model(input_id, segment_id, input_mask, labels=None)

        preds.append(logits.detach().cpu().numpy())
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        # print(preds)
        return(probs[0][0])



def predict_kgbert_multipletails(input1, input2, eval_id=0) :

    eval_features = []
    id = eval_id
    for entity in input2 :
        eval_example = _create_example(input1, entity, DATA_DIR, guid="%s-%s" % ("eval", id))
        id = id + 1
        eval_features.append(convert_example_to_feature(eval_example, label_list, max_seq_length, tokenizer))

    all_input_ids = torch.tensor(
        [f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=len(input2)//4)

    device = torch.device('cpu')
    model.to(device)
    model.eval()

    for input_ids, input_mask, segment_ids in tqdm(eval_dataloader, desc="Testing"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        
        preds = []
        probs = []

        with torch.no_grad():
            logits, *_ = model(input_ids, segment_ids,
                                input_mask, labels=None)

            preds.append(logits.detach().cpu().numpy())
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            print(probs)
    
    # return all probs
    return(probs[:][0])
