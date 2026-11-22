from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import pearsonr
from smolagents import tool
from dotenv import load_dotenv
import os
import matplotlib

# Saving plots without opening GUI windows.
# Setting this is important before importing matplotlib.pyplot.
matplotlib.use("Agg")

from smolagents import CodeAgent, OpenAIServerModel

# --- Task 1: Define Your Tools -----------------------------------------------------------------------------

# Resolve the data folder relative to this script so it works
# regardless of the directory from which Python is launched.
ASSIGNMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = ASSIGNMENT_DIR / "happiness_project"

project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY is missing. Add it to the .env file "
        "at the repository root."
    )

api_key = os.getenv("OPENAI_API_KEY")


df: pd.DataFrame | None = None

COLUMN_NAMES = {
    "Country": "country",
    "Regional indicator": "region",
    "Happiness score": "happiness_score",
    "Ladder score": "happiness_score",
    "GDP per capita": "gdp_per_capita",
}


@tool
def load_happiness_data() -> dict:
    """Load and combine the yearly World Happiness CSV files into memory.

    Read semicolon-separated files with decimal commas, normalize the
    main column names, and extract each year from its filename.
    Store the combined dataset in the shared global DataFrame.

    Returns:
        A dictionary containing shape and columns, or an error dictionary
        if files cannot be loaded or required columns are missing.
    """
    global df

    paths = sorted(DATA_DIR.glob("world_happiness_*.csv"))

    if not paths:
        return {"error": f"No World Happiness CSV files found in {DATA_DIR}."}

    frames = []
    required_columns = {
        "country",
        "region",
        "happiness_score",
        "gdp_per_capita",
    }

    for path in paths:
        try:
            year_df = pd.read_csv(path, sep=";", decimal=",")
            year_df = year_df.rename(columns=COLUMN_NAMES)
            year_df["year"] = int(path.stem.split("_")[-1])
        except (OSError, ValueError, pd.errors.ParserError) as error:
            return {"error": f"Could not load {path.name}: {error}"}

        missing = sorted(required_columns - set(year_df.columns))
        if missing:
            return {
                "error": f"{path.name} is missing required columns: {missing}"
            }

        frames.append(year_df)

    # Updating the shared dataset only after every file loads successfully.
    df = pd.concat(frames, ignore_index=True)

    return {
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
    }


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for one column in the loaded dataset.

    Numeric columns include statistics such as mean, minimum, and maximum.
    Text columns include statistics such as unique count and most common value.

    Args:
        column: Exact name of the column to summarize.

    Returns:
        A dictionary of descriptive statistics, or an error dictionary
        if no data is loaded or the column does not exist.
    """
    if df is None:
        return {"error": "No data is loaded. Call load_happiness_data first."}

    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}

    return df[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute Pearson correlation between two numeric columns.

    Exclude rows where either selected value is missing, keeping the
    remaining observations paired. Round the coefficient and p-value
    to four decimal places.

    Args:
        col1: Exact name of the first numeric column.
        col2: Exact name of the second numeric column.

    Returns:
        A dictionary containing col1, col2, pearson_r, and p_value,
        or an error dictionary for missing data, invalid columns,
        nonnumeric values, or an undefined correlation.
    """
    if df is None:
        return {"error": "No data is loaded. Call load_happiness_data first."}

    missing = [column for column in (col1, col2) if column not in df.columns]
    if missing:
        return {"error": f"Columns not found: {missing}"}

    if not is_numeric_dtype(df[col1]) or not is_numeric_dtype(df[col2]):
        return {"error": "Both columns must be numeric."}

    # Drop missing values together so each pair still represents one row.
    paired = df[[col1, col2]].dropna()
    x = paired.iloc[:, 0]
    y = paired.iloc[:, 1]

    if len(paired) < 2:
        return {"error": "At least two complete observations are required."}

    if x.nunique() < 2 or y.nunique() < 2:
        return {"error": "Correlation is undefined for a constant column."}

    try:
        correlation, p_value = pearsonr(x, y)
    except ValueError as error:
        return {"error": f"Could not compute correlation: {error}"}

    if pd.isna(correlation) or pd.isna(p_value):
        return {"error": "The selected values produced an undefined correlation."}

    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(float(correlation), 4),
        "p_value": round(float(p_value), 4),
    }


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top countries ranked by a numeric column for one year.

    Filter the loaded dataset to the requested year, exclude missing
    ranking values, and sort the selected column in descending order.

    Args:
        column: Numeric column to rank, such as happiness_score.
        year: Year to filter, such as 2024.
        n: Maximum number of countries to return. Must be positive.

    Returns:
        A dictionary with a countries list containing country names and
        their requested column values, or an error dictionary on bad input.
    """
    if df is None:
        return {"error": "No data is loaded. Call load_happiness_data first."}

    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}

    if not is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' must be numeric for ranking."}

    if n < 1:
        return {"error": "n must be a positive integer."}

    year_df = df.loc[df["year"] == year]

    if year_df.empty:
        return {"error": f"No data is available for year {year}."}

    ranked = (
        year_df.dropna(subset=[column])
        .sort_values(column, ascending=False)
        .head(n)
    )

    return {
        "countries": ranked[["country", column]].to_dict(orient="records")
    }


# --- Task 2: Build the Agent -------------------------------------------------------------------------------


def build_agent() -> CodeAgent:
    """Create the World Happiness analysis agent."""
    model = OpenAIServerModel(
        api_key=api_key,
        model_id="gpt-4o-mini",
    )

    system_prompt = """
    You are a data analyst assistant for the World Happiness dataset.

    Use the available tools for loading data, summarizing columns,
    computing correlations, and ranking countries:
    - load_happiness_data
    - summarize_column
    - compute_correlation
    - get_top_n_countries

    Write Python code directly only when the tools are insufficient,
    such as regional aggregation, comparisons across years, or custom plots.

    The tools operate on a shared DataFrame in the host program.
    When supplied, happiness_df is the loaded DataFrame available to your
    generated Python code. Do not assume a variable named df already exists
    inside your code execution environment.

    For regional trends, calculate the mean happiness_score for each
    region and year, then plot one line per region in chronological order.
    Explain that these are unweighted averages of the available countries.

    Save figures inside output_dir using matplotlib savefig.
    Use descriptive titles, axis labels, and legends.
    Call tight_layout before saving and close each figure afterward.
    Do not call show.

    Report correlations as associations, not proof of causation.
    The correlation tool rounds p-values to four decimal places:
    a returned value of 0.0 means rounded to zero, not exactly zero.
    Pooled country-year observations include repeated countries, so avoid
    treating the p-value as conclusive evidence from independent observations.

    Do not invent results or claim that a file was saved without executing
    the code that saves it. Be concise and student-friendly.
    """

    return CodeAgent(
        tools=[
            load_happiness_data,
            summarize_column,
            compute_correlation,
            get_top_n_countries,
        ],
        model=model,
        instructions=system_prompt,
        additional_authorized_imports=[
            "pandas",
            "matplotlib.pyplot",
            "scipy.stats",
        ],
        max_steps=8,
    )


def run_query(agent: CodeAgent, query: str, output_dir: Path):
    """Run a query with conversation history and the current loaded data."""
    additional_args = {
        "outputs": str(output_dir),
    }

    # Tool functions update the host program's global df.
    # Pass the loaded object explicitly to the agent's code environment.
    if df is not None:
        additional_args["happiness_df"] = df

    print(f"\n--- Query: {query} ---")

    response = agent.run(
        query,
        reset=False,
        additional_args=additional_args,
    )

    print("\nResponse:")
    print(response)
    return response


def main() -> None:
    """Run the guided queries and two additional analysis questions."""
    output_dir = ASSIGNMENT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    agent = build_agent()

    # --- Task 3: Run Guided Queries ------------------------------------------------------------------------   

    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        (
            "What is the correlation between gdp_per_capita and "
            "happiness_score? Is it statistically significant?"
        ),
        "Show me the top 5 happiest countries in 2020.",
        (
            "Plot happiness_score over the years as a line chart, "
            "with one line per region. Save the plot to "
            "outputs/happiness_by_region.png."
        ),
    ]

    for index, query in enumerate(queries):
        run_query(agent, query, output_dir)

        # Query 1 must load the host DataFrame so subsequent turns
        # can receive it through additional_args.
        if index == 0 and df is None:
            raise RuntimeError(
                "The first query did not load the dataset. "
                "Check the agent's load_happiness_data result."
            )

    plot_path = output_dir / "happiness_by_region.png"

    if plot_path.is_file() and plot_path.stat().st_size > 0:
        print(f"\nRegional plot found: {plot_path}")
    else:
        raise RuntimeError(
            f"The regional plot was not found or is empty: {plot_path}"
        )

    # Guided query observations:
    # The combined dataset contained 1,502 rows and 11 columns.
    # The mean happiness score was approximately 5.45.
    # The five happiest countries in 2020 were Finland, Denmark,
    # Switzerland, Iceland, and Norway.
    #
    # For Query 5, the agent wrote custom code to calculate mean happiness
    # by year and region, then plotted one line per region and saved it.
    # These are unweighted averages of the available country observations;
    # changes in which countries are included can affect regional trends.


    # --- Task 4: Your Own Questions --------------------------------------------------------------------------

    my_query_1 = (
        "Use get_top_n_countries to show the top 5 countries by "
        "gdp_per_capita in 2024. Include their values."
    )
    run_query(agent, my_query_1, output_dir)

    # My query 1 observation:
    # The agent called get_top_n_countries with gdp_per_capita, year=2024,
    # and n=5. It returned Luxembourg, Ireland, Singapore, United Arab
    # Emirates, and Switzerland, in that order.
    # It used generated Python to invoke the existing tool; no custom
    # ranking logic was needed. These are the dataset's GDP feature
    # values, so I should not describe them as dollar amounts.

    my_query_2 = (
        "Which 5 countries had the largest increase in happiness_score "
        "between 2015 and 2024? Use happiness_df and match countries "
        "by their exact country names, including only countries with "
        "nonmissing scores in both years. Show their 2015 score, "
        "2024 score, and change. Save a labeled bar chart of those "
        "changes to outputs/happiness_improvement.png. "
        "Mention that countries whose names changed may be excluded "
        "and that this comparison does not establish what caused changes."
    )
    run_query(agent, my_query_2, output_dir)

    # My query 2 observation:
    # The agent wrote custom pandas code rather than calling a predefined
    # analysis tool. It selected the 2015 and 2024 records, merged them
    # by exact country name, calculated score changes, and ranked them.
    # It then generated matplotlib code and saved happiness_improvement.png.
    #
    # The largest increases among matched countries were:
    # Ivory Coast: +1.4301
    # Togo:        +1.3741
    # Romania:    +1.3712
    # Guinea:     +1.3633
    # Serbia:     +1.2908
    #
    # The results appeared in the execution log, but the final reply only
    # confirmed the saved chart rather than repeating the requested table.
    # The code also attempted to place a limitations note below the figure,
    # but that note is not visible in the saved image I inspected.
    # Exact-name matching may exclude countries whose names changed.
    # These comparisons describe changes without establishing their causes.

    improvement_path = output_dir / "happiness_improvement.png"
    print(
        "\nImprovement plot found:",
        improvement_path.is_file()
        and improvement_path.stat().st_size > 0,
    )


if __name__ == "__main__":
    main()


# --- Task 5: Reflection ------------------------------------------------------------------------------------
#
# 1. Correlation:
# The agent used compute_correlation and reported Pearson r = 0.6313
# with a p-value rounded to 0.0, calling the result statistically
# significant. This indicates a moderately strong positive linear
# association: higher GDP feature values tend to accompany higher
# happiness scores in this dataset.
# However, its answer did not explain the strength or direction,
# clarify that the p-value was rounded, or discuss the limitations.
# The actual p-value is very small, not necessarily exactly zero.
# Repeated observations of the same countries complicate the usual
# independence assumption, and correlation does not prove causation.
#
# 2. Surprising response:
# I was impressed that the agent matched countries across two years,
# calculated changes, and created a chart without a dedicated tool
# for that analysis.
# I also noticed that it initially tried to access happiness_df before
# that variable was available in its code environment. It recovered
# by calling load_happiness_data again and using the tool's output.
# This showed that an agent can recover from an execution error,
# but successful completion does not mean every step was error-free.
#
# 3. Additional tool:
# I would add compare_country_years(country, start_year, end_year).
# It would validate the requested years, handle missing observations,
# and return changes in happiness and selected related measurements.
# It could also use a documented mapping of country-name variations.
# This would help answer questions such as "How did Mexico's happiness
# score change between 2015 and 2024?" consistently.