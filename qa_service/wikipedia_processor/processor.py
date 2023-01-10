'''
This file was created by ]init[ AG 2022.

Module for Wikipedia Processor.
'''
import bz2
from html import unescape
import logging
from mwparserfromhell import nodes, parse, wikicode
import numpy
import os
import re
from sentence_cleaner_splitter.cleaner_splitter import SentenceSplitClean
from typing import Generator
from .places import read_places
from xml.etree import cElementTree as ET

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def wiki_articles(filepath: str) -> Generator[tuple[str, str, float], None, None]:
    title = None
    ns = None
    wikicode = None
    total_size = os.path.getsize(filepath)
    with open(filepath, 'rb') as f:
        for _, el in ET.iterparse(bz2.BZ2File(f) if filepath.endswith('bz2') else f):
            if '}' in el.tag:
                el.tag = el.tag.split('}', 1)[1]  # strip all namespaces
            if el.tag == 'page':
                if ns == '0' and title and wikicode:
                    yield title, wikicode, float(f.tell() * 100) / total_size
                title = None
                ns = None
                wikicode = None
            elif el.tag == 'title':
                title = el.text
            elif el.tag == 'ns':
                ns = el.text
            elif el.tag == 'text':
                wikicode = el.text
            el.clear()  # else iterparse still builds tree in memory!


def wikicode_texts(wikicode_text) -> Generator[str, None, None]:
    yield from _wikicode_texts(parse(wikicode_text), [])


def _wikicode_texts(wikicode: wikicode.Wikicode, ancestors: list) -> Generator[str, None, None]:
    # wikicode.filter doesn't really work, because we don't know tag ends (nested level) via this API
    for node in wikicode.nodes:
        # most regular and important Text node first
        mode = type(node).__name__
        if isinstance(node, nodes.Text):
            # if node.value.find('Metadaten') != -1:  # for Debugging
            #    pass
            value = node.value
            if ancestors and ancestors[-1] == 'Wikilink':
                if value.startswith('mini|'):
                    values = value.split('|')
                    if values[-1].startswith('alternativtext='):
                        yield f"{values[-2]}: {values[-1][15:]}"
                    else:
                        yield values[-1]
                    continue
                alt_pos = value.find('alt=')
                if alt_pos == 0 or alt_pos > 0 and value[alt_pos - 1] == '|':
                    end_pos = value.find('|', alt_pos + 4)
                    if end_pos >= 0:
                        yield value[end_pos + 1:] + ": "
                    yield value[alt_pos + 4: end_pos]
                    continue
            yield value
        elif isinstance(node, nodes.Argument):
            continue
        elif isinstance(node, nodes.Comment):
            yield f"(Kommentar: {node.contents})"
            continue
        elif isinstance(node, nodes.ExternalLink):
            contents = node.title if node.title else node.url
            yield from _wikicode_texts(contents, ancestors + [mode])
            continue
        elif isinstance(node, nodes.Heading):
            yield ('=' * node.level) + ' '
            yield from _wikicode_texts(node.title, ancestors + [mode])
            yield ' ' + ('=' * node.level)
            continue
        elif isinstance(node, nodes.HTMLEntity):
            # shouldn't happen, because already unescaped before this method
            yield f"&{node.value};"
            continue
        elif isinstance(node, nodes.Tag):
            # if node.tag not in ('b', 'br', 'i', 'li', 'ref', 'small', 'sub', 'table', 'td', 'th', 'tr'):  # 4 debugging
            #     pass
            if node.tag == 'br':
                yield '\n'
            elif node.tag == 'ref':  # Footnote Reference
                yield ' ('
            yield from _wikicode_texts(node.contents, ancestors + [mode + '.' + str(node.tag)])
            if node.tag == 'ref':
                yield ') '
            continue
        elif isinstance(node, nodes.Template):
            if node.name == 'nowrap':
                for param in node.params:
                    yield from _wikicode_texts(param.value, ancestors + [mode + '.' + str(node.name)])
            # no generic handling, template params must be evaluated for special cases
            # yield from _wikicode_texts(node.name, ancestors + [mode + '.' + str(node.name)])
            continue
        elif isinstance(node, nodes.Wikilink):
            contents = node.text if node.text else node.title
            yield from _wikicode_texts(contents, ancestors + [mode])
            continue
        # for child_code in children:
        #     yield from _wikicode_texts(child_code, ancestors + [mode])


sentence_splitter = SentenceSplitClean('deu_Latn', 'default')


def wiki_sentences(
        filepath: str,
        title_pattern: re.Pattern | numpy.ndarray | None = None,
        text_pattern: re.Pattern | None = None,
        filtered: bool | None = None) -> Generator[tuple[str, list[str], float], None, None]:

    for title, wikicode, progress in wiki_articles(filepath):
        # if True:
        #     yield title, [''], progress
        if isinstance(title_pattern, re.Pattern) and not title_pattern.search(title):
            continue
        if isinstance(title_pattern, numpy.ndarray) and (
                title not in title_pattern or '(' in title and not title.split('(')[0].strip() in title_pattern):
            continue
        if isinstance(text_pattern, re.Pattern) and not text_pattern.search(wikicode):
            continue
        # wikicode = str(wikicode).replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        wikicode = unescape(str(wikicode))
        text = ''.join(wikicode_texts(wikicode))
        sentences = []
        for line in text.split('\n'):
            if len(line.split(' ')) <= 10:
                lines = [line]  # don't split any further
            else:
                lines = [sentence for _, _, sentence in sentence_splitter(line)]
            for line in lines:
                if filtered:
                    if len(line.split(' ')) <= 3 and ':' not in line:
                        continue
                    if line.startswith('Kategorie:'):
                        line = f"{title} ist {line[10:]}"
                    elif title not in line:
                        line = f"{title}: {line}"
                sentences.append(line)
        yield title, sentences, progress
