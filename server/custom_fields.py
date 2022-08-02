class Lemma:
    def __init__(self, raw_text, lemma_text, start_offset, end_offset):
        self.raw_text = raw_text
        self.lemma_text = lemma_text
        self.start_offset = start_offset
        self.end_offset = end_offset

    def __str__(self):
        return (
            f"raw_text:{self.raw_text} lemma_text:{self.lemma_text}"
            f"start_offset:{self.start_offset} end_offset:{self.end_offset}"
        )
