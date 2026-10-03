"""Aho-Corasick normalized phrase index; hits are exclusion signals only."""
from collections import deque

class AnchorAutomaton:
    """Aho-Corasick over <=8-word anchors; verify full phrases at a hit."""

    def __init__(self, phrases: list[tuple[str, str, tuple[str, ...]]]):
        self.word_id = {}
        self.transitions = [{}]
        self.failure = [0]
        self.outputs = [[]]
        self.phrases = phrases
        for phrase_id, (_, _, phrase) in enumerate(phrases):
            anchor = phrase[:8]
            node = 0
            for word in anchor:
                token = self.word_id.setdefault(word, len(self.word_id) + 1)
                child = self.transitions[node].get(token)
                if child is None:
                    child = len(self.transitions)
                    self.transitions[node][token] = child
                    self.transitions.append({})
                    self.failure.append(0)
                    self.outputs.append([])
                node = child
            self.outputs[node].append((phrase_id, len(anchor)))
        queue = deque(self.transitions[0].values())
        while queue:
            node = queue.popleft()
            for token, child in self.transitions[node].items():
                queue.append(child)
                fallback = self.failure[node]
                while fallback and token not in self.transitions[fallback]:
                    fallback = self.failure[fallback]
                self.failure[child] = self.transitions[fallback].get(token, 0)
                self.outputs[child].extend(self.outputs[self.failure[child]])

    def find(self, document: tuple[str, ...]):
        node = 0
        seen = set()
        for end, word in enumerate(document):
            token = self.word_id.get(word, 0)
            while node and token not in self.transitions[node]:
                node = self.failure[node]
            node = self.transitions[node].get(token, 0)
            for phrase_id, anchor_len in self.outputs[node]:
                start = end - anchor_len + 1
                task, item_id, phrase = self.phrases[phrase_id]
                if (phrase_id not in seen and
                        document[start:start + len(phrase)] == phrase):
                    seen.add(phrase_id)
                    yield task, item_id, len(phrase)

