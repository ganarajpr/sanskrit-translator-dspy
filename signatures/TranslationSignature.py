import dspy

class Translation(dspy.Signature):
    """Translate to English"""
    lang = dspy.InputField(desc="language to translate from")
    original = dspy.InputField(desc="a sentence for translation")
    english = dspy.OutputField(desc="translation in english")
