'''
This file was created by ]init[ AG 2022.

Module for building a Wikipedia index with sentence embeddings.

First download into folder imports: wget https://dumps.wikimedia.org/dewiki/latest/dewiki-latest-pages-articles-multistream.xml.bz2
'''
from embedding_manager import EmbeddingManager
from wikipedia_processor import read_places, wiki_sentences
import logging
import numpy
from opensearchpy import OpenSearch, helpers
import re
from sentence_transformers import util
import torch

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


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


def _test_similarity():
    embedding_manager = EmbeddingManager()
    embeddings = embedding_manager.embed(
        ['Die Hauptstadt von Deutschland ist Berlin.',
         'In London, der Hauptstadt von GB, wohnen auch viele Deutsche.',
         'Da beißt die Maus keinen Faden ab.'])
    print(f"Embeddings: {embeddings}")


def _test_similarity_wiki():
    embedding_manager = EmbeddingManager()

    for title, sentences, progress in wiki_sentences('imports/Wikipedia-Berlin.xml'):
        log.info(f"##### {title} #####")

        embeddings = embedding_manager.embed(
            [(title if title in sentence else f"{title}: {sentence}") for sentence in sentences])

        query_embedding = embedding_manager.embed(['Welche Flüsse fließen durch Berlin?'])[0]

        cos_scores = util.cos_sim(query_embedding, embeddings)
        top_results = torch.topk(cos_scores[0], k=min(10, len(cos_scores[0])))
        for score, idx in zip(top_results[0], top_results[1]):
            if score >= 0.4:
                log.info(f"(Score: {score:.4f})  {sentences[idx]}")


def _test_parse_wiki():
    places = read_places()
    # title_pattern = re.compile('^Dresden$')
    # Articles: 18870   Sentences: 831349
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


def _test_opensearch_create_index():
    client = get_os_client()
    index = 'wiki-index-s'
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
        response = client.indices.create(index, body=index_body)
        log.info(f"Creating index: {response}")

    # Add a document to the index.
    embedding = numpy.zeros(1024, dtype=float)

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


def _test_opensearch_create_index_nested():
    client = get_os_client()
    index = 'wiki-index'
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
        response = client.indices.create(index, body=index_body)
        log.info(f"Creating index: {response}")

    # Add a document to the index.
    embedding1 = numpy.zeros(1024, dtype=float)
    embedding2 = numpy.zeros(1024, dtype=float)
    embedding2[0] = 1
    embedding3 = numpy.zeros(1024, dtype=float)
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


def _test_opensearch_index_wiki():
    client = get_os_client()
    index = 'wiki-index-s'

    embedding_manager = EmbeddingManager()
    language = 'de'

    places = read_places()
    # title_pattern = re.compile('^Dresden$')
    # Articles: 18839   Sentences: 819134  ->  18.6gb, 22.7 kByte per Entry
    # (Single Sentences, HNSW, nsmlib, cosinesim)
    text_pattern = re.compile('Postleitzahl')
    article_nr = 0
    sentence_nr = 0
    for title, sentences, progress in wiki_sentences(
            'imports/dewiki-latest-pages-articles-multistream.xml', title_pattern=places, text_pattern=text_pattern):
        article_nr += 1
        sentence_nr += len(sentences)
        log.info(f"### {article_nr} / {sentence_nr} ({progress:.2f}%): {title} ###")

        embeddings = embedding_manager.embed([f"{title}: {sentence}" for sentence in sentences])
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


def _test_opensearch_index_wiki_nested():
    client = get_os_client()
    index = 'wiki-index'

    embedding_manager = EmbeddingManager()
    language = 'de'

    places = read_places()
    # title_pattern = re.compile('^Dresden$')
    # Articles: Articles: 18870   Sentences: 831349 -> 19gb, 850219 entries, 22347 Byte per Entry
    # (Single Sentences, HNSW, nsmlib, cosinesim)
    text_pattern = re.compile('Postleitzahl')
    article_nr = 0
    sentence_nr = 0
    for title, sentences, progress in wiki_sentences(
            'imports/dewiki-latest-pages-articles-multistream.xml', title_pattern=places, text_pattern=text_pattern):
        article_nr += 1
        sentence_nr += len(sentences)
        log.info(f"### {article_nr} / {sentence_nr} ({progress:.2f}%): {title} ###")

        embeddings = embedding_manager.embed([f"{title}: {sentence}" for sentence in sentences])
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


def _test_opensearch_search_wiki():
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
    log.info(f"Search results: {response}")


def _test_opensearch_search_wiki_nested():
    client = get_os_client()
    index_name = 'wiki-index'

    embedding_manager = EmbeddingManager()
    embedding = embedding_manager.embed(["Wer ist der Bürgermeister von Berlin?"])[0]

    # Search for the document.
    query = {
        '_source': ['embeddings.sentence'],
        'query': {
            'nested': {
                'path': 'embeddings',
                'query': {
                    'knn': {
                        'embeddings.embedding': {
                            'vector': embedding.cpu().numpy(),
                            'k': 20
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
        index=index_name,
        request_timeout=300
    )
    log.info(f"Search results: {response}")


def _test_opensearch_get_wiki():
    client = get_os_client()
    index_name = 'wiki-index'

    response = client.get(index_name, 'de:Berlin:2', request_timeout=300)
    log.info(f"Search results: {response}")


def main():
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)
    logging.getLogger('opensearch').setLevel(logging.INFO)

    import resource
    logging.info(f"Current limits: {resource.getrlimit(resource.RLIMIT_AS)}")
    # logging.info("Setting limits to 5 GB RAM")
    # resource.setrlimit(resource.RLIMIT_AS, (5000000000, 5000000000))

    _test_opensearch_create_index()
    _test_opensearch_index_wiki()


if __name__ == '__main__':
    main()
