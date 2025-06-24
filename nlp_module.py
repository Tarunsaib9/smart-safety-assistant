import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

def analyze_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    pos_tags = [(token.text, token.pos_) for token in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return {
        "tokens": tokens,
        "pos_tags": pos_tags,
        "entities": entities
    }

def generate_response(text):
    doc = nlp(text)
    entities_found = False

    for ent in doc.ents:
        entities_found = True
        if ent.label_ == "ORG":
            return f"Bruce Power works with many organizations. What more would you like to know about {ent.text}?"
        elif ent.label_ == "PERSON":
            return f"{ent.text} sounds like a key person. Are they involved in the safety system?"
        elif ent.label_ == "GPE":
            return f"{ent.text} is a location. Are you asking about a site or power station?"
        elif ent.label_ == "MONEY":
            return f"That's a financial figure. Is it part of a proposal or cost analysis?"

    if not entities_found:
        text_lower = text.lower()
        if "temperature" in text_lower or "temperatures" in text_lower:
            return "Temperature refers to heat level detected by sensors. Above 85°C might trigger failure prediction."
        elif "vibration" in text_lower:
            return "Vibration is measured for equipment stability. Values above 0.7 may indicate a failure risk."
        elif "failure" in text_lower:
            return "Failures are predicted when sensor thresholds like temperature or vibration are crossed."
        elif "dashboard" in text_lower:
            return "The dashboard shows AI-based failure predictions in real time."

    return "Can you provide more context? I’m here to help with safety, AI, or system insights."
