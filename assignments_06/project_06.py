import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI


def print_sources(response, limit: int | None = None) -> None:
    """Print source filenames, similarity scores, and text previews."""
    sources = response.source_nodes

    if limit is not None:
        sources = sources[:limit]

    if not sources:
        print("No source nodes returned.")
        return

    for number, source in enumerate(sources, start=1):
        score = (
            f"{source.score:.4f}"
            if source.score is not None
            else "Unavailable"
        )

        print(f"\nSource {number}")
        print("Document:", source.node.metadata.get("file_name", "Unknown"))
        print("Similarity score:", score)
        print("Chunk preview:", source.node.get_content()[:200])
        print("-" * 50)


def main() -> None:
    """Build and query an in-memory Groundwork Coffee document index."""
    # --- Step 1: Setup ------------------------------------------------------------------------------------------------

    assignment_dir = Path(__file__).resolve().parent
    project_root = assignment_dir.parent

    load_dotenv(assignment_dir / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is missing. Add it to assignments_06/.env."
        )

    print("API key found in environment.")

    # groundwork_docs is at the repository root.
    docs_dir = project_root / "groundwork_docs"

    assert docs_dir.is_dir(), (
        f"Document directory not found: {docs_dir}. "
        "Place groundwork_docs at the repository root."
    )

    # --- Step 2: Load the Documents ------------------------------------------------------------------------------------------------

    docs = SimpleDirectoryReader(
        input_dir=str(docs_dir),
        required_exts=[".txt"],
    ).load_data()

    print("\n--- Loaded Documents ---")
    print("Document objects loaded:", len(docs))

    for number, document in enumerate(docs, start=1):
        print(
            f"{number}. "
            f"{document.metadata.get('file_name', 'Unknown')}"
        )

    # --- Step 3: Build the Index and Query Engine ----------------------------------------------------------------

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = LlamaOpenAI(model="gpt-4o-mini", temperature=0)

    index = VectorStoreIndex.from_documents(docs)
    query_engine = index.as_query_engine(similarity_top_k=3)

    print("\nIndex built successfully. Ready to answer questions.")

    # --- Step 4: Query the Assistant --------------------------------------------------------------------------------

    questions = [
        "What are Groundwork's hours on weekends?",
        "Do you offer any dairy-free milk options?",
        "How does the loyalty program work?",
        "How did Groundwork Coffee get started?",
        "Do you offer catering or wholesale orders?",
    ]

    for number, question in enumerate(questions, start=1):
        print(f"\n--- Question {number} ---")
        print("Question:", question)

        response = query_engine.query(question)

        print("Answer:", response)
        print_sources(response, limit=1)

        # Step 4 — Response observations:
    # All five answers matched the supplied documents: weekend hours,
    # dairy-free milk options, loyalty rules, founding history, and
    # catering/wholesale minimums were reported accurately.
    # The assistant used confident, direct language without obvious hedging.
    #
    # Some retrieval rankings were surprising. The weekend-hours question
    # ranked our_story.txt first, even though the hours appear in faq.txt.
    # The dairy-free question ranked seasonal_specials.txt first, although
    # the full milk-options list and no-extra-charge policy appear in menu.txt.
    # These answers may have used other nodes among the three retrieved;
    # printing only the first source does not reveal all supporting evidence.
    #
    # The loyalty answer was accurate despite a top similarity score of
    # only 0.2854. A similarity score measures semantic relatedness,
    # not the probability that an answer is correct.
    # Also, a 200-character preview can omit the relevant passage even
    # when that passage exists later in the retrieved chunk.

    # --- Step 5: Find a Failure -------------------------------------------------------------------------------------------------

    difficult_question = (
        "What is the exact total price, including taxes and delivery, "
        "for full-bar catering for 35 people at an Asheville office?"
    )

    # I expect this question to expose missing information.
    # wholesale_catering.txt offers full-bar catering for events of
    # at least 20 people but gives no package prices, tax rates,
    # or delivery charges. It says pricing varies and invites a quote.
    # The assistant should not invent an exact total.

    print("\n--- Step 5: Challenge Question ---")
    print("Question:", difficult_question)

    difficult_response = query_engine.query(difficult_question)

    print("Full answer:", difficult_response)
    print("Retrieved source count:", len(difficult_response.source_nodes))
    print_sources(difficult_response)

    # Step 5 — Challenge observations:
    # I asked for the exact total cost of full-bar catering for 35 people,
    # including taxes and delivery. I expected difficulty because the
    # catering document gives no exact prices or tax/delivery charges.
    #
    # faq.txt ranked first at 0.4856, ahead of wholesale_catering.txt
    # at 0.4415. menu.txt ranked third at 0.3876.
    # The most directly useful document was retrieved, but not ranked first.
    # Regular menu prices were insufficient to calculate a catering quote.
    #
    # The assistant correctly stated that the total could not be determined
    # and directed the user to hello@groundworkcoffee.com with the event
    # details requested by the catering document.
    # Its tone changed from direct factual answers to an explicit statement
    # of missing information. It did not invent a price.
    #
    # This attempt exposed imperfect retrieval ranking and a knowledge gap,
    # but not a demonstrated generation failure: the model handled the
    # missing information appropriately.
    # This successful response does not guarantee appropriate behavior
    # on every unanswerable question.
    #
    # I would improve ranking by testing smaller, topic-focused chunks
    # and evaluating a reranker. I would also require supporting citations
    # and explicitly instruct the model not to invent prices.
    # Displaying all supporting sources would make answers easier to audit.

if __name__ == "__main__":
    main()


# --- Step 6: Reflection ---

# 1. Framework value
# After loading the documents, the central retrieval implementation uses
# three statements: set the embedding model, build the vector index,
# and create the query engine. A fourth statement explicitly selects
# the language model used to generate answers.
# VectorStoreIndex.from_documents() handles chunking, embedding,
# and indexing behind one method call.
# Our keyword retriever required several dozen lines for token cleanup,
# stopword filtering, overlap scoring, sorting, and selecting a result.
# Implementing semantic retrieval manually would additionally require
# managing chunks, embedding requests, vectors, and similarity search.
# A framework reduces repetitive implementation work, but we still
# need to understand its settings and inspect retrieval quality.

# 2. Another valuable use case
# A university could use this approach to help students find answers
# across housing policies, registration instructions, and program handbooks.
# Answers could point students to the relevant document and section,
# reducing time spent searching and repeated questions to staff.
# The index would need updates when policies change, and retrieval
# would need to respect access restrictions on private documents.

# 3. A failure RAG cannot fully prevent
# Even when retrieval returns the correct document, the model can
# misinterpret a condition, omit an exception, or invent a detail.
# For example, it might retrieve the catering policy but fabricate
# a price that the document never provides.
# Source citations, explicit missing-evidence instructions, and
# evaluation reduce this risk without eliminating it.