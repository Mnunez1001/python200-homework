import os
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    """Running the Chat Completions API warmup exercises."""
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is missing. Add it to the .env file "
            "at the repository root."
        )

    # OpenAI() reads OPENAI_API_KEY from the environment.
    client = OpenAI()

    # --- Chat Completions API ----------------------------------------------------------------------------------

    # API Q1
    print("--- API Q1: First Chat Completion ---")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "What is one thing that makes Python "
                    "a good language for beginners?"
                ),
            }
        ],
    )

    print("Response:", response.choices[0].message.content)
    print("Responding model:", response.model)
    print(
        "Total tokens:",
        response.usage.total_tokens if response.usage else "Unavailable",
    )

    # API Q2
    print("\n--- API Q2: Comparing Temperatures ---")

    prompt = "Suggest a creative name for a data engineering consultancy."
    temperatures = [0, 0.7, 1.5]

    for temperature in temperatures:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

        print(f"\nTemperature: {temperature}")
        print("Response:", response.choices[0].message.content)

   
    # Temperatures 0 and 0.7 both returned "DataForge Solutions".
    # At 1.5, the model produced ten suggestions and extra commentary,
    # showing more variation in both wording and response format.
    # Higher temperature does not guarantee a different or better answer.
    # I would use temperature=0 for greater consistency, although it
    # does not guarantee identical results across repeated calls.

    # API Q3
    print("\n--- API Q3: Three Completions in One Call ---")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Give me a one-sentence fun fact about pandas "
                    "(the animal, not the library)."
                ),
            }
        ],
        n=3,
        temperature=1.0,
    )

    for number, choice in enumerate(response.choices, start=1):
        print(f"Completion {number}:", choice.message.content)

    # All three completions described the panda's thumb-like wrist bone.
    # Their wording differed, but n=3 did not guarantee different facts.

    # API Q4
    print("\n--- API Q4: Limiting Output Tokens ---")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Explain how neural networks work.",
            }
        ],
        max_tokens=15,
    )

    print("Limited response:", response.choices[0].message.content)
    print("Finish reason:", response.choices[0].finish_reason)

    if response.usage:
        print("Completion tokens:", response.usage.completion_tokens)

    
    # The explanation stopped after exactly 15 output tokens and was
    # incomplete. finish_reason="length" confirmed it reached the limit.
    # A token limit helps control output size and output-token costs,
    # but setting it too low can cut off useful content.
    # Tokens are pieces of text, so 15 tokens does not mean 15 words.

    
    # --- System Messages and Personas ------------------------------------------------------------------------------------------

    # System Q1
    print("\n--- System Q1: Patient Python Tutor ---")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a patient, encouraging Python tutor. "
                "You always explain things simply and end with "
                "a word of encouragement."
            ),
        },
        {
            "role": "user",
            "content": "I don't understand what a list comprehension is.",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
    )

    print("\nTutor response:")
    print(response.choices[0].message.content)

    print("\n--- System Q1: Concise Technical Reviewer ---")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise, formal Python code reviewer. "
                "Use precise technical language, give one short code "
                "example, and avoid motivational language."
            ),
        },
        {
            "role": "user",
            "content": "I don't understand what a list comprehension is.",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
    )

    print("\nReviewer response:")
    print(response.choices[0].message.content)

    # The patient tutor gave a longer explanation, defined each part
    # of the syntax, compared loops with comprehensions, and ended
    # with encouragement.
    # The technical reviewer gave a short definition and one example,
    # without motivational language.
    # Both explained the same concept, but the system message changed
    # the tone, length, and teaching style.

    # System Q2
    print("\n--- System Q2: Providing Conversation History ---")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "My name is Jordan and I'm learning Python.",
        },
        {
            "role": "assistant",
            "content": (
                "Nice to meet you, Jordan! Python is a great choice. "
                "What would you like to work on?"
            ),
        },
        {
            "role": "user",
            "content": "Can you remind me what my name is?",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
    )

    print("\nResponse with supplied history:")
    print(response.choices[0].message.content)

    # The model correctly answered that the user's name is Jordan
    # because the introduction was included in this request's history.
    # It used the supplied messages rather than remembering another call.
    # The manually supplied assistant reply also provided conversation
    # context, even though it was not generated by an earlier API call.

    # --- Prompt Engineering ------------------------------------------------------------------------------------------

    reviews = [
        "The onboarding process was smooth and the team was welcoming.",
        "The software crashes constantly and support never responds.",
        "Great price, but the documentation is nearly impossible to follow.",
    ]

    task = (
        "Classify the sentiment of the review as positive, negative, "
        "or mixed.\n\n"
    )

    # Prompt Q1 — Zero-Shot
    print("\n--- Prompt Q1: Zero-Shot ---")

    for number, review in enumerate(reviews, start=1):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": task + f'Review: "{review}"',
                }
            ],
            temperature=0,
        )

        print(f"Review {number}:")
        print(response.choices[0].message.content)

    # Prompt Q2 — One-Shot
    print("\n--- Prompt Q2: One-Shot ---")

    one_example = (
        'Example:\n'
        'Review: "Fast shipping but the item arrived damaged."\n'
        'Sentiment: mixed\n\n'
    )

    for number, review in enumerate(reviews, start=1):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": (
                        task + one_example + f'Review: "{review}"'
                    ),
                }
            ],
            temperature=0,
        )

        print(f"Review {number}:")
        print(response.choices[0].message.content)

    # Zero-shot correctly classified all three reviews but used full sentences.
    # One example changed the output to the consistent "Sentiment: label"
    # format. The classifications stayed the same; the format became
    # shorter and matched the example.

    # Prompt Q3 — Few-Shot
    print("\n--- Prompt Q3: Few-Shot ---")

    three_examples = (
        'Examples:\n'
        'Review: "The setup was easy and everything worked perfectly."\n'
        'Sentiment: positive\n\n'
        'Review: "The app is unreliable and the interface is confusing."\n'
        'Sentiment: negative\n\n'
        'Review: "Fast shipping but the item arrived damaged."\n'
        'Sentiment: mixed\n\n'
    )

    for number, review in enumerate(reviews, start=1):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": (
                        task + three_examples + f'Review: "{review}"'
                    ),
                }
            ],
            temperature=0,
        )

        print(f"Review {number}:")
        print(response.choices[0].message.content)

    # Few-shot produced the same labels and format as one-shot in this run.
    # The additional examples did not visibly improve these three results.
    # I would choose zero-shot for simple tasks, one-shot to demonstrate
    # a format, and few-shot to clarify more complicated category boundaries.

    # Prompt Q4 — Step-by-Step Calculation
    print("\n--- Prompt Q4: Salary Calculation ---")

    problem = (
        "A data engineer earns $85,000 per year. She gets a 12% raise, "
        "then 6 months later takes a new job that pays $7,500 more "
        "per year than her post-raise salary. "
        "What is her final annual salary?"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    problem
                    + "\n\nProvide a brief step-by-step calculation "
                    "showing the raise amount, post-raise annual salary, "
                    "and new annual salary. End with a clearly labeled "
                    "'Final answer:'."
                ),
            }
        ],
        temperature=0,
    )

    print("Calculation and final answer:")
    print(response.choices[0].message.content)

    # Breaking the calculation into explicit stages can help avoid
    # applying the raise to the wrong amount or forgetting the job increase.
    # It also makes the arithmetic easier to check, but does not guarantee
    # accuracy. The correct check is 85000 * 1.12 + 7500 = 102700.
    # The six-month interval does not change the final annual salary;
    # it would matter if we were calculating total earnings over time.
    # Q4: The calculation correctly returned $102,700:
    # $85,000 + $10,200 raise + $7,500 job increase.

    # Prompt Q5 — Structured Output
    print("\n--- Prompt Q5: JSON Review Analysis ---")

    review = (
        "I've been using this tool for three months. "
        "It handles large datasets well, "
        "but the UI is clunky and the export options are limited."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Analyze the following review. Return only valid JSON "
                    "with these keys:\n"
                    '- "sentiment": "positive", "negative", or "mixed"\n'
                    '- "confidence": a float from 0 to 1\n'
                    '- "reason": one sentence explaining the sentiment\n'
                    "Do not include Markdown fences or extra commentary.\n\n"
                    f"Review: {review}"
                ),
            }
        ],
        temperature=0,
    )

    raw_response = response.choices[0].message.content or ""
    print("Raw response:")
    print(raw_response)

    try:
        analysis = json.loads(raw_response)
    except json.JSONDecodeError as error:
        print("Could not parse the response as JSON:", error)
        print("Raw response for debugging:", raw_response)
    else:
        if (
            isinstance(analysis, dict)
            and {"sentiment", "confidence", "reason"} <= analysis.keys()
        ):
            print("Sentiment:", analysis["sentiment"])
            print("Confidence:", analysis["confidence"])
            print("Reason:", analysis["reason"])
        else:
            print("Valid JSON, but the expected object keys are missing.")

    # json.loads() checks JSON syntax, not whether field values meet
    # every requirement. A production program would also validate the
    # sentiment, confidence range, and reason type.
    # The response parsed successfully as JSON and identified mixed
    # sentiment, reflecting praise for performance and criticism of the UI
    # and export options. The reported 0.75 confidence is self-reported,
    # not a verified probability of correctness.

    # Prompt Q6 — Delimiters
    print("\n--- Prompt Q6: Instructions Inside Delimiters ---")

    user_text = (
        "First boil a pot of water. Once boiling, add a handful of salt "
        "and the pasta. Cook for 8-10 minutes until al dente. "
        "Drain and toss with your sauce of choice."
    )

    prompt = f"""You will be given text inside triple backticks.
                If it contains step-by-step instructions, rewrite them as a numbered list.
                If it does not contain instructions, respond with exactly: "No steps provided."
                ```{user_text}```
                """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    print("Instruction passage result:")
    print(response.choices[0].message.content)

    prose_text = (
        "The lake reflected the evening sky. "
        "A gentle breeze moved through the trees along the shore."
    )

    prose_prompt = f"""
                    You will be given text inside triple backticks.
                    If it contains step-by-step instructions, rewrite them as a numbered list.
                    If it does not contain instructions, respond with exactly: "No steps provided."
                    ```{prose_text}```
                    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prose_prompt}],
        temperature=0,
    )

    prose_result = (response.choices[0].message.content or "").strip()

    print("\nRegular prose result:")
    print(prose_result)
    print(
        "Exact expected response:",
        prose_result == "No steps provided.",
    )

    # Delimiters help distinguish the text to process from instructions
    # describing how to process it, reducing ambiguity.
    # They are formatting boundaries, not a security guarantee against
    # instructions embedded in untrusted text.
    # Q6: The instruction passage became four numbered steps.
    # The prose passage returned exactly "No steps provided.",
    # so the exact-response check printed True.
    
    # --- Local Models with Ollama ------------------------------------------------------------------------------------------

    # Ollama Q1
    print("\n--- Ollama Q1: OpenAI Comparison Response ---")

    comparison_prompt = (
        "Explain what a large language model is in two sentences."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": comparison_prompt}
        ],
        temperature=0,
    )

    print("OpenAI response:")
    print(response.choices[0].message.content)

    # Ollama response:
    # A large language model is a computer model trained on large amounts
    # of text to recognize language patterns and generate responses.
    # It can perform tasks such as answering questions, summarizing text,
    # and helping write code.
    
    # openAI response:
    # A large language model is an artificial intelligence system designed to understand and 
    # generate human-like text by analyzingvast amounts of written data. It uses deep learning 
    # techniques, particularly neural networks, to predict and produce coherent and contextually 
    # relevant language based on the input it receives.

    # The openAi response is more detailed and uses more technical language, 
    # while the Ollama response is simpler and more concise. 
    # Both responses correctly explain what a large language model is, 
    # but they differ in depth and complexity.

    # A local model can process prompts without sending them to a remote
    # inference service, provided the application keeps processing local.
    # A disadvantage is that it requires local hardware resources and
    # setup; response speed depends on the model and available hardware.


if __name__ == "__main__":
    main()