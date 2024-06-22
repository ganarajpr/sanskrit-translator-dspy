import dspy

class CheckTranslation(dspy.Signature):
    """Check if the Translate to English is correct"""
    lang = dspy.InputField(desc="language to translate from")
    original = dspy.InputField(desc="a sentence for translation")
    translation=dspy.InputField(desc="the translation in english of input from given language")
    is_correct_translation = dspy.OutputField(desc="a boolean True if translation is correct, False otherwise")
