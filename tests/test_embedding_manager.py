"""
This file was created by ]init[ AG 2022.

Tests for Whatever.
"""
import logging
import numpy as np
from qa_service.embedding_manager.manager import EmbeddingManagerOnPrem
import torch

log = logging.getLogger(__name__)


def calculate_cosine_similarity(embedding1, embedding2):
    cos_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return cos_sim.item()


def calculate_cosine_similarity_matrix(embeddings):
    embeddings_normalized = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
    cos_similarity_matrix = torch.mm(embeddings_normalized, embeddings_normalized.t())
    return cos_similarity_matrix


def test_embedding():
    t2tManager = EmbeddingManagerOnPrem()

    sentence_pairs = [
        ("Ich liebe es, Kuchen zu backen.", "Backen ist eine meiner Lieblingsbeschäftigungen."),
        ("Der Hund spielt im Park.", "Der Park ist ein toller Ort zum Spielen für Hunde."),
        ("Der Zug ist pünktlich angekommen.", "Die Ankunft des Zuges erfolgte ohne Verzögerungen."),
        ("Das Wetter ist heute sehr schön.", "Heute scheint die Sonne und es ist angenehm warm."),
        ("Ich habe einen neuen Job gefunden.", "Mir wurde eine neue Arbeitsstelle angeboten."),
        ("Er liest gerne Krimis.", "Er bevorzugt Bücher aus dem Krimi-Genre."),
        ("Sie ist sehr sportlich und fit.", "Sie ist eine begeisterte Sportlerin."),
        ("Die Pflanze benötigt mehr Wasser.", "Die Pflanze ist durstig und muss gegossen werden."),
        (
            "Er hat sich für ein Studium der Informatik entschieden.",
            "Er hat sich entschlossen, Informatik zu studieren.",
        ),
        ("Die Geburtstagsfeier war ein großer Erfolg.", "Die Party zum Geburtstag war sehr gelungen."),
    ]

    flattened_sentences = [sentence for pair in sentence_pairs for sentence in pair]
    embeddings = t2tManager.embed(flattened_sentences)

    for i in range(0, len(embeddings), 2):
        embedding1 = embeddings[i]
        embedding2 = embeddings[i + 1]
        cos_similarity = torch.nn.functional.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item()
        print(f"Satzpaar {(i // 2) + 1} Cos-Embedding-Score: {cos_similarity:.2f}")

    cos_similarity_matrix = calculate_cosine_similarity_matrix(embeddings)
    # Convert the cosine similarity matrix to a NumPy array for better printing
    cos_similarity_matrix_np = cos_similarity_matrix.cpu().numpy()
    np.set_printoptions(linewidth=200, precision=2)
    print(cos_similarity_matrix_np)
