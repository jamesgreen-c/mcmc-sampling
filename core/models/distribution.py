from collections import Counter


class TextPreprocesser:
    """ Preprocess the text provided """

    def __init__(self):
        self.text = ""

    def _preprocess(self, text: str, lower: bool = True):
        text = text.replace("\n", "")
        text = text.lower() if lower else text
        return text

    def _set_text(self, text: str | None):
        self.text = self._preprocess(text) if text is not None else self.text


class StationaryDistribution(TextPreprocesser):
    """

    Take any given text and find the stationary distribution of each character in said text.
    Assumes:
        - text comes from the stationary distribution
    """

    def __init__(self, text: str, valid_symbols: str | list):
        super().__init__()
        self.text = self._preprocess(text=text)
        self.valid_symbols = self._preprocess(text=valid_symbols)
        self.distribution = self.get_distribution()

    def get_distribution(self):
        """ Stationary distribution is the frequency of each letter in the text. """

        counts = Counter(self.text)
        counts = {key: val for key, val in counts.items() if key in self.valid_symbols}
        total = len(self.text)
        return {key: val / total for key, val in counts.items()}


class Transitions(TextPreprocesser):
    """

    Take any given text and find the transition probabilities
    Assumes:
        - text comes from the stationary distribution
    """

    def __init__(self, text: str, valid_symbols: str | list):
        super().__init__()
        self.text = self._preprocess(text)
        self.valid_symbols = self._preprocess(valid_symbols)
        self.transitions = self.get_transition_probabilities()

    def get_transition_probabilities(self):
        """
        Get transition probabilities
            - Count number of times a letter transitions to another let
            - Get normalise probabilities by dividing by the total number of transitions from the given letter
        """

        transitions = {}
        prev_char = self.text[0]

        for char in self.text[1:]:

            if char in self.valid_symbols and prev_char in self.valid_symbols:

                if transitions.get(prev_char) is None:
                    transitions[prev_char] = {}

                if transitions[prev_char].get(char) is None:
                    transitions[prev_char][char] = 0

                transitions[prev_char][char] += 1

            prev_char = char

        for key in transitions:
            key_total = sum(transitions[key].values())
            for key_2 in transitions[key]:
                transitions[key][key_2] = transitions[key][key_2] / key_total

        return transitions
