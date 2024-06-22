from setup import init_ollama_model, get_cosine_similarity, create_examples, import_json 
import glob
import dspy
from dspy.evaluate import Evaluate
import pandas as pd
import random
import os
from modules.TranslationModule import TranslationModule


ollama_model = init_ollama_model()

def get_examples(json_file): 
    json_file = os.path.join(os.path.dirname(__file__),json_file)
    translations = import_json(json_file)
    keys = set(translations['inputs'][0].keys())
    keys.remove('english')
    examples = create_examples(translations['inputs'], keys)
    return examples

#Train and dev samples
translation_examples = get_examples('./examples/translations.json')



def check_similarity(example, pred, trace=None):
    return get_cosine_similarity(example.english,pred.english)


evaluator = Evaluate(devset=translation_examples, num_threads=1, display_progress=True, display_table=0)
tr_zeroshot = TranslationModule()
print('-------------------evaluate zeroshot -------------------------------')
evaluator(tr_zeroshot, metric=check_similarity)

tr_fewshot = TranslationModule()
tr_fewshot.load('translator.json')
print('-------------------evaluate fewshot -------------------------------')
evaluator(tr_fewshot, metric=check_similarity)

tr_mipro = TranslationModule()
tr_mipro.load('translator-mipro.json')
print('-------------------evaluate mipro -------------------------------')
evaluator(tr_mipro, metric=check_similarity)
