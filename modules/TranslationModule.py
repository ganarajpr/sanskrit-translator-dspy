import dspy
from signatures.TranslationSignature import Translation
class TranslationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(Translation)
    
    def forward(self, original, lang):
        prediction = self.generate_answer(original=original, lang=lang)
        return dspy.Prediction(english=prediction.english)

