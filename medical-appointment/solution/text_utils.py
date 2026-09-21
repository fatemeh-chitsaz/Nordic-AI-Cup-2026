"""Conservative normalization plus explicit contradiction checks."""

import re

SMALL_NUMBERS = dict(
    zip(
        (
            "zero one two three four five six seven eight nine ten eleven "
            "twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
        ).split(),
        range(20),
    )
)
TENS = dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10)))
NUMBER_WORDS = set(SMALL_NUMBERS) | set(TENS) | {"hundred", "thousand"}
STOP_WORDS = set(
    (
        "a an the is are was were be been being do does did has have had "
        "should would could will can shall may must of for to in on at by with "
        "from as that this it its there any about patient doctor visit conversation "
        "mention mentioned discuss discussed discussion right correct actually also still"
    ).split()
)
ALIASES = {
    "medicines": "medication",
    "medicine": "medication",
    "drug": "medication",
    "drugs": "medication",
    "tablets": "medication",
    "tablet": "medication",
    "dose": "dosage",
    "doses": "dosage",
    "lungs": "lung",
    "cardiac": "heart",
    "auscultation": "listen",
    "stethoscope": "listen",
    "listened": "listen",
    "listening": "listen",
    "prescriptions": "prescription",
    "renewal": "renew",
    "renewed": "renew",
    "renewing": "renew",
    "pains": "pain",
    "symptoms": "symptom",
    "findings": "finding",
    "examination": "exam",
    "examined": "exam",
    "treatments": "treatment",
    "treated": "treatment",
}

# Only very explicit mutually exclusive terms are used here. This is a guard, not NLP.
OPPOSITE_GROUPS = [
    ({"left"}, {"right"}),
    ({"normal", "negative", "clear"}, {"abnormal", "positive"}),
    ({"increase", "increased", "higher", "raised"}, {"decrease", "decreased", "lower", "reduced"}),
    ({"before"}, {"after"}),
    ({"morning"}, {"evening", "night"}),
    ({"improved", "better"}, {"worse", "worsened"}),
]


def normalize_numbers(text):
    text = text.lower().replace("’", "'")
    text = text.replace("µg", "mcg").replace("μg", "mcg")
    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text)
    tokens = re.findall(r"\d+(?:\.\d+)?|[a-z]+|[^\w\s]", text)
    result = []
    index = 0
    while index < len(tokens):
        if tokens[index] not in NUMBER_WORDS:
            result.append(tokens[index])
            index += 1
            continue
        subtotal, total = 0, 0
        while index < len(tokens):
            word = tokens[index]
            if word in SMALL_NUMBERS:
                subtotal += SMALL_NUMBERS[word]
            elif word in TENS:
                subtotal += TENS[word]
            elif word == "hundred":
                subtotal = max(1, subtotal) * 100
            elif word == "thousand":
                total += max(1, subtotal) * 1000
                subtotal = 0
            elif word == "and" and index + 1 < len(tokens) and tokens[index + 1] in NUMBER_WORDS:
                pass
            else:
                break
            index += 1
        result.append(str(total + subtotal))
    return " ".join(result)


def content_tokens(text):
    tokens = re.findall(r"\d+(?:\.\d+)?|[a-z]+", normalize_numbers(text))
    return [ALIASES.get(token, token) for token in tokens if token not in STOP_WORDS]


def quantities(text):
    units = {
        "mg": ("mass", 1), "milligram": ("mass", 1), "milligrams": ("mass", 1),
        "g": ("mass", 1000), "gram": ("mass", 1000), "grams": ("mass", 1000),
        "mcg": ("mass", 0.001), "microgram": ("mass", 0.001), "micrograms": ("mass", 0.001),
        "day": ("days", 1), "days": ("days", 1), "week": ("days", 7), "weeks": ("days", 7),
        "month": ("months", 1), "months": ("months", 1), "year": ("years", 1), "years": ("years", 1),
        "mm": ("length", 1), "cm": ("length", 10), "millimeter": ("length", 1),
        "millimeters": ("length", 1), "millimetre": ("length", 1), "millimetres": ("length", 1),
        "centimeter": ("length", 10), "centimeters": ("length", 10),
        "centimetre": ("length", 10), "centimetres": ("length", 10),
        "ml": ("volume", 1), "milliliter": ("volume", 1), "milliliters": ("volume", 1),
        "bpm": ("rate", 1),
    }
    matches = re.findall(r"\b(\d+(?:\.\d+)?)\s*([a-z]+)\b", normalize_numbers(text))
    values = set()
    for number, unit in matches:
        if unit in units:
            dimension, multiplier = units[unit]
            values.add((dimension, round(float(number) * multiplier, 6)))
    for year in re.findall(r"\b(?:19|20)\d{2}\b", text):
        values.add(("calendar_year", float(year)))
    return values


def quantity_conflict(question, evidence):
    if re.search(r"\b(not|no|less|more|under|over|least|most|between)\b", question.lower()):
        return False
    observed = quantities(evidence)
    for dimension, value in quantities(question):
        same_dimension = {v for d, v in observed if d == dimension}
        if same_dimension and value not in same_dimension:
            return True
    return False


def polarity_conflict(question, evidence):
    # Negation changes the meaning of polarity words; leave those cases to QA.
    if re.search(r"\b(no|not|never|without|didn.t|isn.t|wasn.t|weren.t)\b", question.lower() + " " + evidence.lower()):
        return False
    q = set(content_tokens(question))
    e = set(content_tokens(evidence))
    for left, right in OPPOSITE_GROUPS:
        if (q & left and e & right) or (q & right and e & left):
            return True
    return False


def explicit_question_conflict(first, second):
    """True only for high-confidence mutually exclusive near-duplicate questions."""
    q1 = set(content_tokens(first))
    q2 = set(content_tokens(second))
    shared = q1 & q2
    base1 = {t for t in q1 if not re.fullmatch(r"\d+(?:\.\d+)?", t)}
    base2 = {t for t in q2 if not re.fullmatch(r"\d+(?:\.\d+)?", t)}
    similarity = len(base1 & base2) / max(1, len(base1 | base2))
    if similarity < 0.35 or len(shared) < 1:
        return False

    a, b = quantities(first), quantities(second)
    for dim1, value1 in a:
        for dim2, value2 in b:
            if dim1 == dim2 and value1 != value2:
                return True

    for left, right in OPPOSITE_GROUPS:
        if (q1 & left and q2 & right) or (q1 & right and q2 & left):
            return True
    return False
