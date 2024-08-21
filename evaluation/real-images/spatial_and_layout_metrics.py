import re

# from .metrics_base import MultipleChoiceMetric


class SpatialAndLayoutReasoningMetric():
    """This class is a metric that requires a correct prediction to be only one of the valid multiple choice answers."""

    def __init__(self):
        super().__init__()

    def __evaluate__(self, answer_text, target_text, target_options, is_valid):
        if not is_valid:
            return "none"

        # count which options appear
        options_count = {}
        for option in target_options:

            # make sure only matches whole words by includeing a word boundary "\b" in the pattern
            pattern = "\\b{phrase}\\b".format(phrase=option)

            matches = re.findall(pattern, answer_text, flags=re.IGNORECASE)  # search
            options_count[option] = len(matches)  # count

        total_count = sum(options_count.values())

        # correct if only the right answer appears once,
        # incorrect if only a wrong answer appears,
        # none if there are mutiple answers or nothing matches
        return (
            "correct"
            if (total_count == 1 and options_count[target_text] == 1)
            else "incorrect" if (total_count == 1) else "none"
        )
