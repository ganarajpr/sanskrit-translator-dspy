from setup import create_examples, init_ollama_model, import_json, get_cosine_similarity
import os
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, MIPRO
from modules.TranslationModule import TranslationModule
from signatures.CheckTranslationSignature import CheckTranslation
import dspy

ollama_model = init_ollama_model()

def get_examples(json_file): 
    json_file = os.path.join(os.path.dirname(__file__),json_file)
    translations = import_json(json_file)
    keys = set(translations['inputs'][0].keys())
    keys.remove('english')
    examples = create_examples(translations['inputs'], keys)
    return examples

translation_examples = get_examples('./examples/translations.json')

print(translation_examples);


def check_translation(example, pred, trace=None):
    lang, original, translation = example.lang, example.original, pred.english
    check_translation_accuracy = dspy.ChainOfThought(CheckTranslation)
    result = check_translation_accuracy(lang=lang, original=original, translation=translation)
    is_accurate = result.is_correct_translation.lower() == 'true'
    return is_accurate 

def check_similarity(example, pred, trace=None):
    return get_cosine_similarity(example.english,pred.english)


# teleprompter = BootstrapFewShotWithRandomSearch(
#     max_bootstrapped_demos=5,
#     max_labeled_demos=5,
#     num_candidate_programs=5,
#     num_threads=6,
#     metric=check_similarity)
num_new_prompts_generated = 5
prompt_generation_temperature = 0.6
teleprompter = MIPRO(prompt_model=ollama_model, task_model=ollama_model, metric=check_similarity,
                      num_candidates=num_new_prompts_generated, init_temperature=prompt_generation_temperature)

kwargs = dict(num_threads=4, display_progress=True, display_table=0)

compiled_program_optimized_bayesian_signature = teleprompter.compile(TranslationModule(), trainset=translation_examples, num_trials=30, max_bootstrapped_demos=3, max_labeled_demos=5, eval_kwargs=kwargs)
# compiled_translator = teleprompter.compile(TranslationModule(), trainset=translation_examples)
compiled_program_optimized_bayesian_signature.save('translator-mipro.json')
print(ollama_model.inspect_history(1))
my_question = "सख्युः सखिसमा वाह्याद्गामियानासनादयः ज्ञातेः स्वसृदुहित्रात्मजाग्रजावरजादयः"
pred = compiled_program_optimized_bayesian_signature(original=my_question, lang='sanskrit')
print("----------------Final Prediction------------------------")
print(pred)
