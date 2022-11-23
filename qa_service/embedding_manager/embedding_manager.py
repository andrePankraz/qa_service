'''
This file was created by ]init[ AG 2022.

Module for Sentence Embedding Models.
'''
import logging
import mwparserfromhell
import numpy
from opensearchpy import OpenSearch, helpers
import os
import re
from sentence_cleaner_splitter.cleaner_splitter import SentenceSplitClean
from sentence_transformers import SentenceTransformer, util
import threading
from timeit import default_timer as timer
import torch
from typing import Generator
from xml.etree import cElementTree as ET

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class EmbeddingManager:

    lock = threading.Lock()

    def __new__(cls):
        # Singleton!
        if not hasattr(cls, 'instance'):
            cls.instance = super(EmbeddingManager, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        if hasattr(self, 'sentence_splitters'):
            return
        with EmbeddingManager.lock:
            # Load model Sentence Embedding
            # Max sequence length is 512 -> embedding is 1024 dimensional
            model_id = 'Sahajtomar/German-semantic'
            model_folder = os.environ.get('MODEL_FOLDER', '/opt/speech_service/models/')
            device = 'cpu'
            if torch.cuda.is_available():
                log.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                mem_info = torch.cuda.mem_get_info(0)
                vram = round(mem_info[1] / 1024**3, 1)
                log.info(
                    f"VRAM available: {round(mem_info[0]/1024**3,1)} GB out of {vram} GB")
                if (vram >= 4):
                    device = 'cuda:0'
                    # model_id = 'facebook/nllb-200-3.3B' if vram >= 32 else 'facebook/nllb-200-distilled-1.3B' if vram >= 12 else 'facebook/nllb-200-distilled-600M'
            log.info(f"Loading model {model_id!r} in folder {model_folder!r}...")
            self.model = SentenceTransformer(model_id, device=device, cache_folder=model_folder)
            log.info("...done.")
            if device != 'cpu':
                log.info(f"VRAM left: {round(torch.cuda.mem_get_info(0)[0]/1024**3,1)} GB")

    def embed(self, sentences: list[str]) -> torch.Tensor:
        log.debug(f"Embedding {len(sentences)} sentences...")
        start = timer()
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        log.debug(f"...done in {timer() - start:.3f}s")
        assert isinstance(embeddings, torch.Tensor)
        return embeddings


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


def get_os_client() -> OpenSearch:
    # see https://opensearch.org/docs/latest/clients/python/
    host = 'host.docker.internal'  # 'localhost'
    port = 9200
    auth = ('admin', 'admin')  # For testing only. Don't store credentials in code.
    # ca_certs_path = '/full/path/to/root-ca.pem'  # Provide a CA bundle if you use intermediate CAs with your root CA.

    # Optional client certificates if you don't want to use HTTP basic authentication.
    # client_cert_path = '/full/path/to/client.pem'
    # client_key_path = '/full/path/to/client-key.pem'

    # Create the client with SSL/TLS enabled, but hostname verification disabled.
    # Test: curl -XGET https://localhost:9200 -u 'admin:admin' --insecure
    client = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_compress=True,  # enables gzip compression for request bodies
        http_auth=auth,
        # client_cert = client_cert_path,
        # client_key = client_key_path,
        use_ssl=True,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False
        # ca_certs=ca_certs_path
    )
    log.info(f"Ping: {client.ping()}")
    return client


def read_places() -> numpy.ndarray:
    from pandas import read_csv, concat
    data = read_csv('uploads/4681_geodatendeutschland_1001_20210723.csv', sep=',', header='infer')
    return concat([data['KREIS_NAME'], data['GEMEINDE_NAME'], data['ORT_NAME']]).unique()


def test_embedding():
    embedding_manager = EmbeddingManager()
    embeddings = embedding_manager.embed(
        ['Die Hauptstadt von Deutschland ist Berlin.',
         'In London, der Hauptstadt von GB, wohnen auch viele Deutsche.',
         'Da beißt die Maus keinen Faden ab.'])
    print(f"Embeddings: {embeddings}")


def test_parse_wiki():
    places = read_places()
    # title_pattern = re.compile('^Dresden$')
    # Articles: 18839   Sentences: 819134
    text_pattern = re.compile('Postleitzahl')
    article_nr = 0
    sentence_nr = 0
    for title, sentences, progress in wiki_sentences(
            'uploads/dewiki-latest-pages-articles-multistream.xml', title_pattern=places, text_pattern=text_pattern):
        article_nr += 1
        sentence_nr += len(sentences)
        log.info(f"### {article_nr} / {sentence_nr} ({progress:.2f}%): {title} ###")
        # print(f"{sentences}")
    log.info(f"Articles: {article_nr}   Sentences: {sentence_nr}")


def test_wiki():
    embedding_manager = EmbeddingManager()

    for title, sentences, progress in wiki_sentences('uploads/Wikipedia-20221120102219.xml'):
        log.info(f"##### {title} #####")

        embeddings = embedding_manager.embed(
            [(title if title in sentence else f"{title}: {sentence}") for sentence in sentences])

        query_embedding = embedding_manager.embed(['Welche Flüsse fließen durch Berlin?'])[0]

        cos_scores = util.cos_sim(query_embedding, embeddings)
        top_results = torch.topk(cos_scores[0], k=min(10, len(cos_scores[0])))
        for score, idx in zip(top_results[0], top_results[1]):
            if score >= 0.4:
                log.info(f"(Score: {score:.4f})  {sentences[idx]}")


def test_opensearch_create_index():
    client = get_os_client()

    index_name = 'wiki-index'

    if client.indices.exists(index_name):
        # Delete the index.
        response = client.indices.delete(
            index=index_name
        )

        print('\nDeleting index:')
        print(response)

    if not client.indices.exists(index_name):
        # Create an index with non-default settings.
        # See: https://opensearch.org/docs/latest/search-plugins/knn/approximate-knn/
        # See:
        # https://aws.amazon.com/de/blogs/big-data/choose-the-k-nn-algorithm-for-your-billion-scale-use-case-with-opensearch/

        # Default with Replica 0: 25 kB per document entry - Why?
        # FAISS: 1.1 * (4 * d + 8 * m) * num_vectors
        # FAISS: 1.1 * (4 * 1024 + 8 * 16) * num_vectors
        index_body = {
            'settings': {
                'index': {
                    'number_of_shards': 1,
                    'number_of_replicas': 0,
                    'knn': True,
                    'knn.algo_param.ef_search': 256  # only nmslib
                }
            },
            'mappings': {
                'properties': {
                    'sentence': {
                        'type': 'text',
                        'index': False
                    },
                    'embedding': {
                        'type': 'knn_vector',
                        'dimension': 1024,
                        'method': {
                            'name': 'hnsw',
                            'engine': 'nmslib',
                            'space_type': 'cosinesimil',
                            'parameters': {
                                'm': 16,
                                'ef_construction': 256
                            }
                        }
                    }
                }
            }
        }

        response = client.indices.create(index_name, body=index_body)
        print('\nCreating index:')
        print(response)

    # Add a document to the index.
    import numpy as np
    embedding = np.zeros(1024, dtype=float)

    language = 'de'
    title = 'Berlin'
    sentence = 'Testsatz'
    sentence_idx = 1

    id = f"{language}:{title}:{sentence_idx}"
    document = {
        'sentence': sentence,
        'embedding': embedding
    }

    response = client.index(
        index=index_name,
        body=document,
        id=id,
        refresh=True
    )

    print('\nAdding document:')
    print(response)

    # Search for the document.
    q = 'Testsatz'
    query = {
        'size': 5,
        'query': {
            'knn': {
                'embedding': {
                    'vector': embedding,
                    'k': 2
                }
            }
        }
    }

    response = client.search(
        body=query,
        index=index_name
    )
    print('\nSearch results:')
    print(response)


def test_opensearch_create_multi_index():
    client = get_os_client()

    index_name = 'wiki-multi-index'

    if client.indices.exists(index_name):
        # Delete the index.
        response = client.indices.delete(
            index=index_name
        )

        print('\nDeleting index:')
        print(response)

    if not client.indices.exists(index_name):
        # Create an index with non-default settings.
        # See: https://opensearch.org/docs/latest/search-plugins/knn/knn-index/
        # See:
        # https://aws.amazon.com/de/blogs/big-data/choose-the-k-nn-algorithm-for-your-billion-scale-use-case-with-opensearch/

        # Default with Replica 0: 25 kB per document entry - Why?
        # FAISS: 1.1 * (4 * d + 8 * m) * num_vectors
        # FAISS: 1.1 * (4 * 1024 + 8 * 16) * num_vectors
        index_body = {
            'settings': {
                'index': {
                    'number_of_shards': 1,
                    'number_of_replicas': 0,
                    'knn': True,
                    'knn.algo_param.ef_search': 256  # only nmslib
                }
            },
            'mappings': {
                'properties': {
                    'embeddings': {
                        'type': 'nested',
                        'properties': {
                            'sentence': {
                                'type': 'text',
                                'index': False
                            },
                            'embedding': {
                                'type': 'knn_vector',
                                'dimension': 1024,
                                'method': {
                                    'name': 'hnsw',
                                    'engine': 'nmslib',
                                    'space_type': 'cosinesimil',
                                    'parameters': {
                                        'm': 16,
                                        'ef_construction': 256
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        response = client.indices.create(index_name, body=index_body)
        print('\nCreating index:')
        print(response)

    # Add a document to the index.
    import numpy as np
    embedding1 = np.zeros(1024, dtype=float)
    embedding2 = np.zeros(1024, dtype=float)
    embedding2[0] = 1
    embedding3 = np.zeros(1024, dtype=float)
    embedding3[0] = -1

    sentence1 = 'Testsatz1'
    sentence2 = 'Testsatz2'
    sentence3 = 'Testsatz3'

    language = 'de'
    title = 'Berlin'

    id = f"{language}:{title}"
    document = {
        'embeddings': [
            {
                'sentence': sentence1,
                'embedding': embedding1
            },
            {
                'sentence': sentence2,
                'embedding': embedding2
            },
            {
                'sentence': sentence3,
                'embedding': embedding3
            }
        ]
    }

    response = client.index(
        index=index_name,
        body=document,
        id=id,
        refresh=True
    )

    print('\nAdding document:')
    print(response)

    # Search for the document.
    q = 'Testsatz'
    query = {
        'query': {
            'nested': {
                'path': 'embeddings',
                'query': {
                    'knn': {
                        'embeddings.embedding': {
                            'vector': embedding2,
                            'k': 2
                        }
                    }
                },
                'inner_hits': {
                }
            }
        }
    }

    response = client.search(
        body=query,
        index=index_name
    )
    print('\nSearch results:')
    print(response)


def test_opensearch_index_wiki():
    client = get_os_client()
    index_name = 'wiki-index'

    embedding_manager = EmbeddingManager()
    language = 'de'
    # Articles: 18839   Sentences: 819134  ->  18.6gb, 22.7 kByte per Entry
    # (Single Sentences, HNSW, nsmlib, cosinesim)

    places = read_places()
    # title_pattern = re.compile('^Dresden$')
    # Articles: 18839   Sentences: 819134
    text_pattern = re.compile('Postleitzahl')
    article_nr = 0
    sentence_nr = 0
    for title, sentences, progress in wiki_sentences(
            'uploads/dewiki-latest-pages-articles-multistream.xml', title_pattern=places, text_pattern=text_pattern):
        article_nr += 1
        sentence_nr += len(sentences)
        log.info(f"### {article_nr} / {sentence_nr} ({progress:.2f}%): {title} ###")

        embeddings = embedding_manager.embed([f"{title}: {sentence}" for sentence in sentences])
        actions = []
        for sentence_idx, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            id = f"{language}:{title}:{sentence_idx}"
            document = {
                'sentence': sentence,
                'embedding': embedding.cpu().numpy()
            }
            actions.append({
                '_op_type': 'index',
                '_index': index_name,
                '_id': id,
                '_source': document
            })
            # TODO see for bulk: https://stackoverflow.com/questions/72632710/using-opensearch-python-bulk-api-to-insert-data-to-multiple-indices
            # response = client.index(
            #     index=index_name,
            #     body=document,
            #     id=id,
            #     refresh=True
            # )

            # print('\nAdding document:')
            # print(response)
        success, errors = helpers.bulk(
            client,
            actions,
            request_timeout=300,  # default is 10, not enough for bulk
            max_retries=30  # min_backoff 2s, doubling each retry, max_backoff 600s
        )
        log.info(f"...Successfully indexed {success} sentences.")
        if errors:
            log.info(f"   Errors: {errors}")
    log.info(f"Articles: {article_nr}   Sentences: {sentence_nr}")


def test_opensearch_search_wiki():
    client = get_os_client()
    index_name = 'wiki-index'

    embedding_manager = EmbeddingManager()
    embedding = embedding_manager.embed(["Wer ist der Bürgermeister von Berlin?"])[0]

    # Search for the document.
    query = {
        '_source': ['sentence'],
        'query': {
            'knn': {
                'embedding': {
                    'vector': embedding.cpu().numpy(),
                    'k': 50
                }
            }
        }
    }

    response = client.search(
        body=query,
        index=index_name,
        request_timeout=300
    )
    print('\nSearch results:')
    print(response)


def test_opensearch_get_wiki():
    client = get_os_client()
    index_name = 'wiki-index'

    response = client.get(index_name, 'de:Berlin:2', request_timeout=300)
    print('\nSearch results:')
    print(response)


def _main_debug():
    '''
    Just for debugging.
    '''
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)
    logging.getLogger('opensearch').setLevel(logging.INFO)

    import resource
    logging.info(f"Current limits: {resource.getrlimit(resource.RLIMIT_AS)}")
    # logging.info("Setting limits to 5 GB RAM")
    # resource.setrlimit(resource.RLIMIT_AS, (5000000000, 5000000000))

    test_opensearch_search_wiki()


if __name__ == '__main__':
    '''
    Just for debugging.
    '''
    _main_debug()
