"""NLP engine: NLTK preprocessing + TF-IDF intent recognition."""
import json
import random
import re
from datetime import datetime
from pathlib import Path

from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent

# These NLTK components need no downloaded corpora, so the bot works offline.
_tokenizer = RegexpTokenizer(r"[a-z0-9]+")
_stemmer = PorterStemmer()

# Small stop-word list. Question words (what, how, who, you, me...) are
# deliberately kept because they carry intent.
_stop = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "about", "it", "its", "this", "that", "these", "those", "be", "am",
    "are", "was", "were", "been", "my", "your", "i", "so", "very", "just",
    "some", "any", "please", "s", "t",
}


def preprocess(text: str) -> str:
    tokens = _tokenizer.tokenize(text.lower())
    tokens = [_stemmer.stem(t) for t in tokens if t not in _stop]
    return " ".join(tokens)


class Chatbot:
    def __init__(self, intents_path=BASE_DIR / "intents.json", threshold=0.65):
        with open(intents_path, encoding="utf-8") as f:
            data = json.load(f)
        self.intents = {i["tag"]: i for i in data["intents"]}
        self.fallback = data["fallback"]
        self.threshold = threshold

        self.patterns, self.tags = [], []
        for intent in data["intents"]:
            for p in intent["patterns"]:
                self.patterns.append(preprocess(p))
                self.tags.append(intent["tag"])

        # word 1-2 grams help short phrases like "tf idf"
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.patterns)

    def predict_intent(self, message: str):
        cleaned = preprocess(message)
        if not cleaned:
            return None, 0.0
        vec = self.vectorizer.transform([cleaned])
        scores = cosine_similarity(vec, self.matrix)[0]
        best = int(scores.argmax())
        return self.tags[best], float(scores[best])

    def respond(self, message: str) -> dict:
        tag, score = self.predict_intent(message)
        if tag is None or score < self.threshold:
            return {"reply": random.choice(self.fallback), "intent": "fallback", "confidence": round(score, 2)}
        reply = random.choice(self.intents[tag]["responses"])
        now = datetime.now()
        reply = reply.format(time=now.strftime("%I:%M %p"), date=now.strftime("%A, %d %B %Y"))
        return {"reply": reply, "intent": tag, "confidence": round(score, 2)}


if __name__ == "__main__":
    bot = Chatbot()
    print("Chatbot ready. Type 'quit' to exit.")
    while True:
        msg = input("You: ").strip()
        if msg.lower() in {"quit", "exit"}:
            break
        r = bot.respond(msg)
        print(f"Bot [{r['intent']} {r['confidence']}]: {r['reply']}")
