import html
from html.parser import HTMLParser
import string


class IndexedHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.convert_charrefs = False
        self.indices = []
        self.text = []

    def handle_charref(self, name):
        pos = self.getpos()[1]
        self.indices.append(pos)
        spec = f"&#{name};"
        self.text.append(html.unescape(spec))

    def handle_entityref(self, name):
        pos = self.getpos()[1]
        self.indices.append(pos)
        spec = f"&{name};"
        self.text.append(html.unescape(spec))

    def handle_data(self, data):
        self.text.append(data)
        pos = self.getpos()[1]

        for i in range(len(data)):
            self.indices.append(i + pos)

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.indices = []
        self.text = []

    def get_parsed_text(self):
        return "".join(self.text)
