from dotenv import load_dotenv
import os
import string

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is missing from the environment.")


# --- RAG Concepts -----------------------------------------------------------------------------------

# Concepts Q1

# Scenario A — RAG
# RAG can retrieve relevant passages from the internal policy PDFs
# and provide them as context for an answer. Updating the document
# index each quarter keeps the reference material current without
# retraining the language model.

# Scenario B — Fine-tuning
# Fine-tuning is the best fit for teaching a specialized brand voice
# using the startup's 3,000 writing examples. A curated set of
# instruction-and-response examples can demonstrate the desired
# minimalist style consistently.

# Scenario C — Prompt engineering
# Include the two-page report directly in the prompt along with the
# analyst's question and instructions to answer from the supplied text.
# For one short document and a one-time task, a retrieval index would
# add unnecessary complexity.


# Concepts Q2

# A confidently wrong answer can discourage verification because its
# authoritative tone makes unsupported information sound established.
# An answer that acknowledges uncertainty signals that the user
# should check the evidence before relying on it.
# For example, an assistant could invent an internal security policy
# saying employees may upload customer records to a public service.
# An employee who trusts that confident answer could expose private
# information. Fluent wording is not evidence that an answer is correct.


# Concepts Q3 — Correct RAG pipeline order

# Document preparation / indexing:

# 1. Extract text from source documents.
#    Read PDFs or other files and obtain their text for processing.

# 2. Split text into chunks.
#    Divide the extracted text into smaller passages that can be
#    retrieved individually.

# 3. Convert text chunks into embeddings.
#    Represent each chunk as a vector and store it with its text
#    and source metadata in a searchable index.

# Query processing / answering:

# 4. Receive the user's query.
#    Accept the question that the system needs to answer.

# 5. Embed the user's query.
#    Convert the question into a vector using the same embedding
#    model and compatible settings used for the document chunks.

# 6. Retrieve the most relevant chunks.
#    Search the index for chunks whose embeddings are most similar
#    to the query embedding.

# 7. Inject retrieved chunks into the prompt.
#    Combine the question, retrieved context, and answering instructions.

# 8. Generate a response from the LLM.
#    Ask the language model to answer using the retrieved evidence
#    and acknowledge when that evidence is insufficient.


print("--- RAG Concepts ---")
print("Q1: Approach choices are documented in the comments.")
print("Q2: Hallucination risks are documented in the comments.")
print("Q3: The ordered RAG pipeline is documented in the comments.")

# --- Keyword RAG ------------------------------------------------------------------------------------

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "your", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]



# Keyword Q1
documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

print("\n--- Keyword Q1: Weekend Hours ---")

query = "What are your hours on weekends?"
matches = simple_keyword_retrieval(query, documents, verbose=True)

print("Selected document:", matches[0][0])

# The function selects hours.txt because "weekends" appears in both
# the query and that document. Its overlap score is 1.
# The filename itself is not searched, so the word "hours" in
# hours.txt does not contribute to the score.


# Keyword Q2
print("\n--- Keyword Q2: Without Caffeine ---")

query = "Do you have anything without caffeine?"
matches = simple_keyword_retrieval(query, documents, verbose=True)

print("Selected document:", matches[0][0])

# The function returns "None found" because no remaining query tokens
# match words in any document.
# It cannot connect "without caffeine" to beverage or menu information.
# Semantic retrieval using embeddings could identify menu.txt as
# related to the request even without matching the exact words.
# However, the menu does not explicitly identify caffeine-free options.
# Better retrieval would not justify inventing an answer about which
# drinks are caffeine-free; the system should acknowledge that gap.

# Keyword Q3

# My prediction before running:
# since I don't see any document that mentions "rewards" or "signup", 
# I predict that the function will return "None found" because there are no overlapping 
# keywords between the query and the documents.

print("\n--- Keyword Q3: Rewards Signup ---")

query = "How do I sign up for rewards?"
matches = simple_keyword_retrieval(query, documents, verbose=True)

print("Selected document:", matches[0][0])

# My observation after running:
# My prediction was correct; the function returned "None found" because 
# there are no overlapping keywords between the query and the documents.

# --- Semantic RAG Concepts --------------------------------------------------------------------------------

# Semantic Q1

# 1. A vector embedding represents text as a list of numbers learned
#    by a model. These numbers help compare texts by patterns associated
#    with their meaning rather than only their exact words.

# 2. The chunk scoring 0.85 is more similar to the query than the one
#    scoring 0.30, assuming both use the same embedding model and metric.
#    Cosine similarity compares vector directions; 0.85 does not mean
#    an 85% probability that the chunk contains the correct answer.

# 3. Semantic search can connect different expressions of related ideas,
#    such as "employee perks" and "staff benefits", because their learned
#    representations can be similar despite different vocabulary.


# Semantic Q2

# | Feature                 | Keyword RAG                    | Semantic RAG                       |
# |-------------------------|--------------------------------|------------------------------------|
# | What is compared?       | Exact word overlap             | Query and text embedding vectors   |
# | What is retrieved?      | Full document                  | Relevant text chunks / nodes       |
# | Can it handle synonyms? | No                             | Often, through semantic similarity |
# | Storage format          | Plain text dictionary          | Vector index plus text and metadata|
# | Relevance score         | Number of overlapping keywords | Vector similarity, such as cosine  |

# --- LlamaIndex -----------------------------------------------------------------------------------------------------

# LlamaIndex Q1
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# The embedding model supports retrieval.
# This separate language model writes answers from retrieved context.
Settings.llm = LlamaOpenAI(model="gpt-4o-mini", temperature=0)

assignment_dir = Path(__file__).resolve().parent.parent

# Adjust this path to the location of the lesson folder.
pdf_dir = (
    assignment_dir / "brightleaf_pdfs"
)

print("\n--- LlamaIndex Q1: BrightLeaf Solar ---")
print("PDF directory:", pdf_dir.resolve())

if not pdf_dir.is_dir():
    raise FileNotFoundError(
        f"BrightLeaf PDF folder not found: {pdf_dir.resolve()}. "
        "Update pdf_dir to point to your lesson's brightleaf_pdfs folder."
    )

docs = SimpleDirectoryReader(
    input_dir=str(pdf_dir),
    required_exts=[".pdf"],
).load_data()

print("Loaded document objects:", len(docs))

# Creates chunks, embeds them, and builds an in-memory vector index.
index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for question in questions:
    print("\nQuestion:", question)

    response = query_engine.query(question)
    print("Answer:", response)

    print("Retrieved source nodes:", len(response.source_nodes))

    for number, node_with_score in enumerate(
        response.source_nodes,
        start=1,
    ):
        score = node_with_score.score
        score_text = f"{score:.4f}" if score is not None else "Unavailable"

        node = node_with_score.node
        snippet = node.get_content()[:150]

        print(f"\nSource {number}")
        print("File:", node.metadata.get("file_name", "Unknown"))
        print("Page:", node.metadata.get("page_label", "Unknown"))
        print("Similarity score:", score_text)
        print("Text snippet:", snippet)
        print("-" * 40)

# Employee benefits query:
# employee_benefits.pdf ranked first with a similarity score of 0.7620.
# Its snippet introduces the employee benefits program, making it
# directly relevant to the question.
# The answer sounds confident and specific: it names insurance,
# retirement contributions, parental leave, and development support,
# without hedging with phrases such as "I'm not sure."
# partnerships.pdf (0.5966) and mission_statement.pdf (0.5651) were
# less obviously relevant based on their displayed snippets.
# Their retrieval was somewhat unexpected, although content elsewhere
# in those chunks could explain the connection.
# The first 150 characters alone cannot verify every benefit claimed
# in the answer; I would check the full source text before relying on it.


# Security policies query:
# security_policy.pdf ranked first with a similarity score of 0.6730.
# Its snippet discusses network defenses and access to critical systems,
# which directly relates to the question.
# The answer is confident, detailed, and declarative, describing specific
# controls and standards without explicitly acknowledging uncertainty.
# employee_benefits.pdf (0.6135) and partnerships.pdf (0.5915) were
# unexpected secondary results because their snippets do not directly
# describe security policies.
# Shared company context may contribute to their similarity, but the
# snippets alone do not establish why they ranked this highly.
# I would verify detailed claims, such as credential rotation periods
# and encryption standards, against the full security policy.


# Overall retrieval observation:
# similarity_top_k=3 returns the three highest-ranked available chunks,
# even when the second and third results are less relevant.
# Similarity measures semantic relatedness, not factual correctness.
# A confident answer and a relevant first source do not guarantee that
# every statement in the generated response is supported.

# PDF extraction observation:
# The reader reported startxref warnings but continued loading documents.
# The security snippet contained "multif", suggesting an extraction
# artifact. Imperfect extracted text can affect retrieval and generation,
# so successful execution alone does not establish clean PDF extraction.

# LlamaIndex Q2 — Compare Retrieval Sizes

print("\n--- LlamaIndex Q2: Comparing top_k=1 and top_k=5 ---")

comparison_question = "What employee benefits does BrightLeaf offer?"

for top_k in [1, 5]:
    comparison_engine = index.as_query_engine(similarity_top_k=top_k)
    comparison_response = comparison_engine.query(comparison_question)

    print(f"\nsimilarity_top_k={top_k}")
    print("Question:", comparison_question)
    print("Answer:", comparison_response)

    for number, source in enumerate(
        comparison_response.source_nodes, start=1
    ):
        score = (
            f"{source.score:.4f}"
            if source.score is not None
            else "Unavailable"
        )
        print(
            f"Source {number}: "
            f"{source.node.metadata.get('file_name', 'Unknown')} "
            f"| Score: {score}"
        )

# LlamaIndex Q2 — Observations:
#
# With top_k=1, only employee_benefits.pdf was retrieved, scoring 0.7621.
# The answer organized benefits into four sections and included specific
# details such as the $600 annual wellness allowance and the 5% 401(k) match.
#
# With top_k=5, the benefits document still ranked first at 0.7620,
# followed by partnerships, mission statement, earnings, and security.
# The answer covered similar benefit categories but omitted the $600
# wellness allowance. More retrieved context did not produce more detail.
#
# More context is not always better: extra chunks may add useful evidence,
# but they can also introduce unrelated information and increase token use.
# In this run, top_k=1 already supplied the source needed for the question.
# This single comparison does not prove that extra context caused the
# omission; generated answers can vary between calls.

# LlamaIndex Q3 — A Difficult Query

print("\n--- LlamaIndex Q3: Information That May Be Missing ---")

difficult_question = (
    "What is BrightLeaf's exact monthly reimbursement amount "
    "for employees' pet insurance?"
)

# Prediction before running:
# I expect the documents may not specify a monthly pet-insurance amount.
# The system should acknowledge missing evidence instead of inventing
# a number, even if it retrieves general employee-benefit information.

difficult_response = query_engine.query(difficult_question)

print("Question:", difficult_question)
print("Answer:", difficult_response)

for number, source in enumerate(
    difficult_response.source_nodes, start=1
):
    score = (
        f"{source.score:.4f}"
        if source.score is not None
        else "Unavailable"
    )

    print(f"\nRetrieved chunk {number}")
    print("File:", source.node.metadata.get("file_name", "Unknown"))
    print("Similarity score:", score)
    print("Full chunk text:")
    print(source.node.get_content())
    print("-" * 50)

# LlamaIndex Q3 — Observations:
#
# I expected the documents might not contain an exact monthly
# pet-insurance reimbursement amount.
# The answer acknowledged the gap: "The provided information does not
# specify any details regarding pet insurance reimbursement."
#
# employee_benefits.pdf ranked first at 0.5536, but its full text described
# general benefits and a $600 annual wellness allowance, not pet insurance.
# The model appropriately avoided treating that allowance as pet coverage
# or dividing it into an invented monthly pet-insurance reimbursement.
# partnerships.pdf (0.4126) and mission_statement.pdf (0.4026) did not
# provide the requested information either.
#
# I would make the missing-evidence instruction explicit and require
# supporting citations for specific benefit amounts.
# I would also test more unanswerable questions and consider reranking
# or metadata filters to reduce unrelated retrieval.
# Similarity alone cannot establish that an exact requested fact exists.


# LlamaIndex Q4 — Evaluate Responses

from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)

print("\n--- LlamaIndex Q4: LLM-Based Evaluation ---")

judge_llm = LlamaOpenAI(model="gpt-4o-mini", temperature=0)

faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

evaluation_questions = [
    "What employee benefits does BrightLeaf offer?",
    "What were the names of the astronauts on BrightLeaf's 2024 Moon mission?",
]

for question in evaluation_questions:
    # Using the original top_k=3 engine from Q1.
    evaluated_response = query_engine.query(question)

    faithfulness_result = faithfulness_evaluator.evaluate_response(
        response=evaluated_response,
    )

    relevancy_result = relevancy_evaluator.evaluate_response(
        query=question,
        response=evaluated_response,
    )

    print("\nQuestion:", question)
    print("Answer:", evaluated_response)

    print("Faithfulness score:", faithfulness_result.score)
    print("Faithfulness passing:", faithfulness_result.passing)
    print("Faithfulness feedback:", faithfulness_result.feedback)

    print("Relevancy score:", relevancy_result.score)
    print("Relevancy passing:", relevancy_result.passing)
    print("Relevancy feedback:", relevancy_result.feedback)


# LlamaIndex Q4 — Evaluation Reflection:
#
# The employee-benefits answer received faithfulness=1.0 and relevancy=1.0.
# Both evaluators returned passing=True with feedback "YES".
# The Moon-mission answer received 0.0 for both metrics, with
# passing=False and feedback "NO".
#
# Faithfulness=1.0 means the judge accepted the answer as supported by
# the retrieved context. A score of 0.0 means it did not accept that
# support under its evaluation prompt.
# These are binary judgments, not percentages of correct statements
# or guarantees that an answer is true.
#
# Relevancy evaluates whether the response and retrieved context address
# the question. Faithfulness focuses on support from the retrieved context.
# An answer can repeat source material faithfully while missing the question.
#
# The benefits query had relevant supporting information.
# The Moon-mission query requested information absent from the supplied
# context, and the model appropriately acknowledged that absence.
# Its failing scores therefore should not automatically be interpreted
# as hallucination or poor behavior.
# One possible explanation is that the judges did not accept an
# absence-of-information statement as supported or sufficient to answer
# the question. Their brief YES/NO feedback does not establish why.
#
# LLM-as-a-judge uses a language model to assess responses against
# criteria such as source support and relevance.
# This accommodates valid answers with different wording and does not
# require an exact reference answer, unlike simple exact-match accuracy.
# However, judges can misclassify appropriate responses, so I would
# supplement these scores with human review and a separate check for
# appropriate abstention when the documents do not contain an answer.
