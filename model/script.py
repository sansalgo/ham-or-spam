import json


def load_csv(filepath):
    data = []
    with open(filepath, 'r', encoding='latin-1') as f:
        lines = f.readlines()
        for line in lines[1:]:  # skip header
            parts = line.strip().split(',', 2)  # split only first 2 commas
            if len(parts) >= 2:
                label = parts[0].strip().lower().replace('"', '')
                message = parts[1].strip().strip('"')
                data.append((message, label))
    return data


def tokenize(text):
    return text.lower().split()


def train_naive_bayes(data):
    vocab = set()
    word_counts = {"spam": {}, "ham": {}}
    class_counts = {"spam": 0, "ham": 0}

    for message, label in data:
        class_counts[label] += 1
        words = tokenize(message)
        for word in words:
            vocab.add(word)
            word_counts[label][word] = word_counts[label].get(word, 0) + 1

    total_messages = len(data)
    priors = {c: class_counts[c] / total_messages for c in ["spam", "ham"]}

    likelihood = {"spam": {}, "ham": {}}
    for label in ["spam", "ham"]:
        total_words = sum(word_counts[label].values())
        for word in vocab:
            count = word_counts[label].get(word, 0)
            likelihood[label][word] = (count + 1) / (total_words + len(vocab))

    return vocab, priors, likelihood


def save_model(vocab, priors, likelihood, filename="model.json"):
    model = {
        "vocab": list(vocab),
        "priors": priors,
        "likelihood": likelihood
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)


if __name__ == "__main__":
    filepath = "spam.csv"
    training_data = load_csv(filepath)
    vocab, priors, likelihood = train_naive_bayes(training_data)
    save_model(vocab, priors, likelihood)
    print("✅ Model trained and saved to model.json")
