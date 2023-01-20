'''
This file was created by ]init[ AG 2022.

Module for building a Wikipedia index with sentence embeddings.

First download into folder imports: wget https://dumps.wikimedia.org/dewiki/latest/dewiki-latest-pages-articles-multistream.xml.bz2
'''
from embedding_manager import EmbeddingManager
import logging
import numpy
from opensearchpy import OpenSearch, helpers
import os
from qa_manager import QaManager
import re
from sentence_transformers import util
import torch
from wikipedia_processor import download, read_places, wiki_sentences

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

imports_folder = os.environ.get('IMPORTS_FOLDER', '/opt/qa_service/imports/')


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


def _test_similarity(embedding_manager: EmbeddingManager):
    embeddings = embedding_manager.embed(
        ['Was ist die Hauptstadt von Deutschland?',
         'Von welchem Land ist Berlin die Hauptstadt?',
         'Die Hauptstadt von Deutschland ist Berlin.',
         'Berlin ist die Hauptstadt von Deutschland.',
         'Die Hauptstadt von Italien ist Rom.',
         'Rom ist die Hauptstadt von Italien',
         'In Rom, der Hauptstadt von Italien, wohnen auch viele Deutsche.',
         'In Amerika gibt es eine kleine Stadt namens Berlin.',
         'Da beißt die Maus keinen Faden ab.'])

    # torch.set_printoptions(profile="full")
    torch.set_printoptions(linewidth=200)

    print(f"Embeddings Shape: {embeddings.shape}")
    # print(torch.mm(embeddings, embeddings.T))
    print(f"Similarity:\n{util.cos_sim(embeddings, embeddings)}")


def _test_similarity_wiki(embedding_manager: EmbeddingManager, question: str, wikipage: str, number: int = 10):
    download(f"https://de.wikipedia.org/wiki/Spezial:Exportieren/{wikipage}", f"{imports_folder}{wikipage}.xml")

    for title, sentences, _ in wiki_sentences(f"{imports_folder}{wikipage}.xml", filtered=True):
        log.info(f"##### {title} #####")

        with open(f"{imports_folder}{wikipage}.txt", "wt") as f:
            print('\n'.join(sentences), file=f)

        query_embedding = embedding_manager.embed([question])[0]
        sentence_embeddings = embedding_manager.embed(sentences)

        cos_scores = util.cos_sim(query_embedding, sentence_embeddings)
        top_results = torch.topk(cos_scores[0], k=min(number, len(cos_scores[0])))
        for score, idx in zip(top_results[0], top_results[1]):
            if score >= 0.4:
                log.info(f"(Score: {score:.4f})  {sentences[idx]}")


def _test_download_wiki():
    download('https://dumps.wikimedia.org/dewiki/latest/dewiki-latest-pages-articles-multistream.xml.bz2',
             imports_folder + 'dewiki-latest-pages-articles-multistream.xml.bz2', 'Downloading Wikipedia-Export')


def _test_parse_wiki():
    places = read_places()
    # title_pattern = re.compile('^Dresden$')
    # Articles: 28283   Sentences: 2595783
    text_pattern = re.compile('Postleitzahl|PLZ')
    article_nr = 0
    sentence_nr = 0
    for title, sentences, progress in wiki_sentences(
            imports_folder + 'dewiki-latest-pages-articles-multistream.xml.bz2', title_pattern=places, text_pattern=text_pattern, filtered=True):
        article_nr += 1
        sentence_nr += len(sentences)
        log.info(f"### {article_nr} / {sentence_nr} ({progress:.2f}%): {title} ###")
        # print(f"{sentences}")
    log.info(f"Articles: {article_nr}   Sentences: {sentence_nr}")


def _test_opensearch_create_index(embedding_manager: EmbeddingManager, index: str = 'wiki_index'):
    client = get_os_client()
    if client.indices.exists(index):
        response = client.indices.delete(index)
        log.info(f"Deleting index: {response}")
    if not client.indices.exists(index):
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
                        'dimension': embedding_manager.get_dimensions(),
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
        response = client.indices.create(index, body=index_body)
        log.info(f"Creating index: {response}")

    # Add a document to the index.
    embedding = numpy.zeros(embedding_manager.get_dimensions(), dtype=float)

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
        index=index,
        body=document,
        id=id,
        refresh=True
    )
    log.info(f"Adding document: {response}")

    # Search for the document.
    query = {
        'query': {
            'knn': {
                'embedding': {
                    'vector': embedding,
                    'k': 2
                }
            }
        }
    }
    response = client.search(body=query, index=index)
    log.info(f"Search results: {response}")


def _test_opensearch_create_index_nested(embedding_manager: EmbeddingManager, index: str = 'wiki_index'):
    client = get_os_client()
    if client.indices.exists(index):
        response = client.indices.delete(index)
        log.info(f"Deleting index: {response}")
    if not client.indices.exists(index):
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
                                'dimension': embedding_manager.get_dimensions(),
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
        response = client.indices.create(index, body=index_body)
        log.info(f"Creating index: {response}")

    # Add a document to the index.
    embedding1 = numpy.zeros(embedding_manager.get_dimensions(), dtype=float)
    embedding2 = numpy.zeros(embedding_manager.get_dimensions(), dtype=float)
    embedding2[0] = 1
    embedding3 = numpy.zeros(embedding_manager.get_dimensions(), dtype=float)
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
        index=index,
        body=document,
        id=id,
        refresh=True
    )
    log.info(f"Adding document: {response}")

    # Search for the document.
    query = {
        'size': 5,
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
    response = client.search(body=query, index=index)
    log.info(f"Search results: {response}")


def _test_opensearch_index_wiki(embedding_manager: EmbeddingManager, index: str = 'wiki_index'):
    client = get_os_client()
    language = 'de'
    places = read_places()
    # title_pattern = re.compile('^Dresden$')
    # Articles: 18944   Sentences: 1507697  ->  18.9gb, 831349 entries, 22734 Byte per Entry
    # (Single Sentences, HNSW, nsmlib, cosinesim)
    text_pattern = re.compile('Postleitzahl|PLZ')
    article_nr = 0
    sentence_nr = 0
    for title, sentences, progress in wiki_sentences(
            imports_folder + 'dewiki-latest-pages-articles-multistream.xml.bz2', title_pattern=places, text_pattern=text_pattern, filtered=True):
        article_nr += 1
        sentence_nr += len(sentences)
        log.info(f"### {article_nr} / {sentence_nr} ({progress:.2f}%): {title} ###")

        embeddings = embedding_manager.embed(sentences)
        actions = []
        for sentence_idx, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            document = {
                'sentence': sentence,
                'embedding': embedding.cpu().numpy()
            }
            # See for bulk:
            # https://stackoverflow.com/questions/72632710/using-opensearch-python-bulk-api-to-insert-data-to-multiple-indices
            actions.append({
                '_op_type': 'index',
                '_index': index,
                '_id': f"{language}:{title}:{sentence_idx}",
                '_source': document
            })
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


def _test_opensearch_index_wiki_nested(embedding_manager: EmbeddingManager, index: str = 'wiki_index'):
    client = get_os_client()
    language = 'de'
    places = read_places()
    # title_pattern = re.compile('^Dresden$')
    # Articles: 18944   Sentences: 1507697 -> 19gb, 850219 entries, 22347 Byte per Entry
    # (Nested Sentences, HNSW, nsmlib, cosinesim)
    text_pattern = re.compile('Postleitzahl|PLZ')
    article_nr = 0
    sentence_nr = 0
    for title, sentences, progress in wiki_sentences(
            imports_folder + 'dewiki-latest-pages-articles-multistream.xml.bz2', title_pattern=places, text_pattern=text_pattern, filtered=True):
        article_nr += 1
        sentence_nr += len(sentences)
        log.info(f"### {article_nr} / {sentence_nr} ({progress:.2f}%): {title} ###")

        embeddings = embedding_manager.embed(sentences)
        nested = []
        for sentence, embedding in zip(sentences, embeddings):
            nested.append({
                'sentence': sentence,
                'embedding': embedding.cpu().numpy()
            })
        document = {'embeddings': nested}
        response = client.index(
            index=index,
            body=document,
            id=f"{language}:{title}",
            request_timeout=300  # default is 10
        )
        log.info(f"Adding document: {response}")
        log.info(f"...Successfully indexed {len(nested)} sentences.")
    log.info(f"Articles: {article_nr}   Sentences: {sentence_nr}")


def _test_opensearch_search_wiki(
        embedding_manager: EmbeddingManager,
        question: str,
        number: int = 10,
        index: str = 'wiki_index'):
    client = get_os_client()
    embedding = embedding_manager.embed([question])[0]
    # Search for the document.
    query = {
        'size': number,
        '_source': ['sentence'],
        'query': {
            'knn': {
                'embedding': {
                    'vector': embedding.cpu().numpy(),
                    'k': number
                }
            }
        }
    }
    response = client.search(
        body=query,
        index=index,
        request_timeout=300
    )
    # log.info(f"Search results: {response}")
    hits = [[hit['_score'], hit['_id'], hit['_source']['sentence']] for hit in response['hits']['hits']]
    sentences = [f"{score:.4f}: {id:<20} {sentence}" for (score, id, sentence) in hits]
    out = '\n'.join(sentences)
    log.info(f"Search results:\n{out}")


def _test_opensearch_search_wiki_nested(embedding_manager: EmbeddingManager, question: str,
                                        number: int = 10,
                                        index: str = 'wiki_index'):
    client = get_os_client()
    embedding = embedding_manager.embed([question])[0]
    # Search for the document.
    query = {
        'size': number,
        '_source': ['embeddings.sentence'],
        'query': {
            'nested': {
                'path': 'embeddings',
                'query': {
                    'knn': {
                        'embeddings.embedding': {
                            'vector': embedding.cpu().numpy(),
                            'k': number
                        }
                    }
                },
                'inner_hits': {
                    '_source': False
                }
            }
        }
    }
    response = client.search(
        body=query,
        index=index,
        request_timeout=300
    )
    log.info(f"Search results: {response}")


def _test_opensearch_get_wiki(id: str,
                              index: str = 'wiki_index'):
    client = get_os_client()
    response = client.get(index, id, request_timeout=300)
    log.info(f"Search results: {response}")


def _test_answer():
    qa_manager = QaManager()
    print(qa_manager.answer('Wer ist der Bürgermeister von Dresden?', 'Der Bürgermeister von Dresden ist Max Mustermann.'))


def main():
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)
    logging.getLogger('opensearch').setLevel(logging.INFO)

    # import resource
    # logging.info(f"Current limits: {resource.getrlimit(resource.RLIMIT_AS)}")
    # logging.info("Setting limits to 5 GB RAM")
    # resource.setrlimit(resource.RLIMIT_AS, (5000000000, 5000000000))

    embedding_manager = EmbeddingManager()

    # _test_similarity(embedding_manager)
    _test_similarity_wiki(embedding_manager, 'Wie heißt der Bürgermeister von Dresden?', 'Dresden', 50)

    # _test_download_wiki()
    # _test_parse_wiki()

    # _test_opensearch_create_index(embedding_manager)
    # _test_opensearch_index_wiki(embedding_manager)
    # _test_opensearch_search_wiki(embedding_manager, 'Wie heißt der Bürgermeister von Berlin?', 100)

    # _test_opensearch_get_wiki('de:Dresden:1')

    # _test_answer()

if __name__ == '__main__':
    main()
