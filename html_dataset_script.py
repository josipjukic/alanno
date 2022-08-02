from bs4 import BeautifulSoup
import os
import sys
import json
import re
from bs4 import BeautifulSoup


def prepend_space(html):
    regex = r'<.+?>'
    html = re.sub(regex, r' \g<0>', html)
    return html

def remove_tags(html, tags=['potpisnik']):
    for tag in tags:
        re.sub(f'(<{tag}.*?>).*?(</{tag}.*?>)', '', html)
    return html

def remove_attrs(soup, whitelist=tuple()):
    for tag in soup.findAll(True):
        for attr in [attr for attr in tag.attrs if attr not in whitelist]:
            del tag[attr]
    return soup

def replace_tags(soup, tags):
    for tag in soup.findAll(re.compile('|'.join(tags))):
        tag.name = 'p'
    return soup

def replace_divs(soup):
    for div in soup.findAll(re.compile('div.+?')):
        div.name = 'p'
    return soup


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Please provide path to directory with HTML files.')
        sys.exit(0)

    print(f'Parsing documents in directory: {sys.argv[1]}')

    data = []
    for (root, directories, files) in os.walk(sys.argv[1]):
        for file in files:
            fp = os.path.join(root, file)
            with open(fp, 'r') as f:
                html = f.read()
                # Prepend empty space to each tag
                # (to ensure correct indexing during the annotation process)
                html = prepend_space(html)
                # html = remove_tags(html)
                soup = BeautifulSoup(html, 'html.parser')
                soup = remove_attrs(soup, whitelist=('style'))
                soup = replace_divs(soup)
                data.append({'html': str(soup.body)})

    new_file = './data/html_documents.json'
    print(f'Data saved to: {new_file}')
    json.dump(data, open(new_file, 'w'))



