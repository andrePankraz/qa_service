'''
This file was created by ]init[ AG 2022.

Module for Wikipedia Processor.
'''
import logging
import mwparserfromhell
import numpy
import os
import re
from places import read_places
from sentence_cleaner_splitter.cleaner_splitter import SentenceSplitClean
from typing import Generator
from xml.etree import cElementTree as ET

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def wiki_articles(filepath: str) -> Generator[tuple[str, str, float], None, None]:
    title = None
    ns = None
    wikicode = None
    total_size = os.path.getsize(filepath)
    with open(filepath, 'r') as f:
        for event, el in ET.iterparse(f):
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
    wikicode = mwparserfromhell.parse(wikicode_text)
    for m in wikicode.ifilter(False):
        if isinstance(m, mwparserfromhell.nodes.text.Text):
            if '|' in m.value:  # skip 'mini|...|'
                prefix, delim, last = m.value.rpartition('|')
                m.value = last if delim else prefix
            yield m.value
        elif isinstance(m, mwparserfromhell.nodes.heading.Heading):
            yield from wikicode_texts(m.title)
        elif isinstance(m, mwparserfromhell.nodes.wikilink.Wikilink):
            link_text = m.title if not m.text else m.text
            yield from wikicode_texts(link_text)
        elif isinstance(m, mwparserfromhell.nodes.external_link.ExternalLink):
            yield from wikicode_texts(m.title)
        elif isinstance(m, mwparserfromhell.nodes.tag.Tag):
            if m.tag == 'ref':
                pass
            elif m.has('mode') and m.get('mode').value == 'packed':
                # special mode for elif m.tag == 'gallery':
                for lines in str(m.contents).split('\n'):
                    if '|' in lines:
                        prefix, delim, last = lines.partition('|')
                        l = '- ' + last if delim else prefix
                    yield from wikicode_texts(lines + '\n')
            else:
                yield from wikicode_texts(m.contents)
        elif isinstance(m, mwparserfromhell.nodes.html_entity.HTMLEntity):
            yield from wikicode_texts(m.value)
        elif isinstance(m, mwparserfromhell.nodes.template.Template):
            if m.name.startswith('Infobox Gemeinde in Deutschland'):
                for param in m.params:
                    yield param.name.strip() + ' ist '
                    yield from wikicode_texts(param.value.strip() + '\n')
        elif isinstance(m, mwparserfromhell.nodes.comment.Comment):
            pass
        else:
            yield f"#?#{type(m)} {m}#?#"


sentence_splitter = SentenceSplitClean('deu_Latn', 'default')


def wiki_sentences(
        filepath: str,
        title_pattern: re.Pattern | numpy.ndarray | None = None,
        text_pattern: re.Pattern | None = None) -> Generator[tuple[str, list[str], float], None, None]:

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
        wikicode = str(wikicode).replace('&nbsp;', ' ').replace('&#8239;', ' ')
        text = ''.join(wikicode_texts(wikicode))
        sentences = []
        for line in text.split('\n'):
            for _, _, sentence in sentence_splitter(line):
                if sentence.startswith('Kategorie:'):
                    sentences.append(f"{title} ist {sentence[10:]}")
                    continue
                l = len(sentence.split(' '))
                if l == 2 and ':' in sentence[:-1]:
                    # Simple table entries like "Postleitzahl: 55555\n",
                    # but not "Aufzählung (Auswahl):\n"
                    sentences.append(' ist '.join(sentence.split(':')))
                    continue
                if l >= 3:
                    # ignore single or double words (e.g. headlines)
                    sentences.append(sentence)
        yield title, sentences, progress


def test_parse_wiki():
    places = read_places()
    # title_pattern = re.compile('^Dresden$')
    # Articles: 18839   Sentences: 819134
    text_pattern = re.compile('Postleitzahl')
    article_nr = 0
    sentence_nr = 0
    for title, sentences, progress in wiki_sentences(
            'imports/dewiki-latest-pages-articles-multistream.xml', title_pattern=places, text_pattern=text_pattern):
        article_nr += 1
        sentence_nr += len(sentences)
        log.info(f"### {article_nr} / {sentence_nr} ({progress:.2f}%): {title} ###")
        # print(f"{sentences}")
    log.info(f"Articles: {article_nr}   Sentences: {sentence_nr}")


def _main_debug():
    '''
    Just for debugging.
    '''
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)
    logging.getLogger('opensearch').setLevel(logging.INFO)

    import resource
    log.info(f"Current limits: {resource.getrlimit(resource.RLIMIT_AS)}")
    log.info("Setting limits to 5 GB RAM")
    resource.setrlimit(resource.RLIMIT_AS, (5000000000, 5000000000))

    test_parse_wiki()


if __name__ == '__main__':
    '''
    Just for debugging.
    '''
    _main_debug()
