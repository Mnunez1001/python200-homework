import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

# smolagents imports
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent


project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY is missing. Add it to the .env file "
        "at the repository root."
    )

# OpenAI() reads OPENAI_API_KEY from the environment.
client = OpenAI()

# --- Lesson 02: Tool Definitions and the ReAct Loop ---------------------------------------------------------

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

# Q1

celsius_tool_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": (
            "Convert a Celsius temperature to Fahrenheit "
            "and return a formatted string."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "The temperature in degrees Celsius.",
                }
            },
            "required": ["celsius"],
        },
    },
}

print("\n--- Q1: Direct Temperature Conversions ---")
print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))

# Q1 observation:
# Direct calls returned 0°C = 32°F, 100°C = 212°F, and -40°C = -40°F.
# These calculations ran locally without any API requests.
# The tool schema describes the function and its numeric celsius
# parameter to the model; it does not execute the function itself.

#Q2----------------------------------------------------------------------------------------------------------

def get_current_time() -> str:
    '''Return the current local time as a formatted string.'''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Returns the current local time as a string.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
    }
]
print('Tools list defined with one tool: get_current_time')

def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time.'''
    
    # Step 1: start the conversation with system and user messages
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',  # model chooses whether to use a tool
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    # Record what the model said so far
    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            # In this example we only have one tool: get_current_time
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            elif function_name == 'celsius_to_fahrenheit':
                arguments = json.loads(tool_call.function.arguments)
                tool_result = celsius_to_fahrenheit(arguments['celsius'])
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            # Print for debugging so we can see what happened
            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")

    # If there were no tool calls, the first response was already the final answer
    return first_message.content or ''

# Q2

# Prediction before running:
# I expect no tool call because the only advertised tool is
# get_current_time, which cannot convert temperatures.
# Defining celsius_to_fahrenheit in Python does not make it available
# to the model unless its schema is included in the tools list.
# The model may answer the conversion from its general knowledge.
#
# I expect one completion API call if it answers without a tool.
# In this starter implementation, requesting a tool would instead
# cause a second completion API call after the Python tool executes.

print("\n--- Q2: Agent With Only the Time Tool ---")
response_q2 = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print("Q2 response:", response_q2)

# Q2 observation:
# My prediction was correct: no tool was called.
# The first response had tool_calls=None, and the program printed
# "No tools needed....".
# Only get_current_time was advertised, so the model answered the
# conversion directly instead of using our Python conversion function.
# This execution made one completion API call and correctly returned 212°F.

# Q3----------------------------------------------------------------------------------------------------------

# Extending the advertised tools only after the original Q2 test has run.
tools.append(celsius_tool_schema)

print("\n--- Q3: Agent With Both Tools ---")
print("Available tools:", [tool["function"]["name"] for tool in tools])

response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)

# Response A observation:
# The model requested celsius_to_fahrenheit with {"celsius": 37}.
# Our dispatcher parsed the JSON arguments and executed the Python function,
# which returned "37°C is 98.6°F".
# The program sent that tool result back in a second completion API call,
# and the model produced the final sentence.
# This execution used two completion API calls and one local tool execution.

response_b = run_agent("What is the boiling point of water in plain English?")
print("Response B:", response_b)

# Response B observation:
# No tool was called: tool_calls=None appeared in the first response.
# The model answered the general boiling-point question directly,
# including the sea-level qualification and both temperature units.
# This execution used one completion API call.
# With tool_choice="auto", an available tool is not necessarily used;
# the model decides whether to request it for the particular question.

# --- Lesson 03: Multi-Tool Agents ----------------------------------------------------------------------
RESOURCES_DIR = Path("resources")

class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # --- Tools (public methods) --------------------------------------

    def list_csv_files(self) -> dict:
        """List available CSV files in resources/.

        Returns:A dict with a "files" list, or a message if none are found.
        """
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.

        filename can be "bike_commute" or "bike_commute.csv".
        """
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """
        Return column names for the currently loaded CSV.
        """
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """
        Simple summary for a single column using pandas.describe().
        """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    # Q4
    def compute_correlation(self, col1: str, col2: str) -> dict:
        """
        Compute the Pearson correlation between two loaded columns.

        Return the column names, correlation coefficient, and p-value.
        Return an error dictionary if no CSV is loaded or a column is missing.
        """
        error = self._ensure_loaded()
        if error:
            return error

        missing = [
            column
            for column in (col1, col2)
            if column not in self.df.columns
        ]
        if missing:
            return {"error": f"These columns are not in the data: {missing}"}

        correlation, p_value = pearsonr(
            self.df[col1],
            self.df[col2],
        )

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(correlation), 4),
            "p_value": round(float(p_value), 4),
        }

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """
        Plot from the active CSV.
    
        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """
        error = self._ensure_loaded()
        if error:
            return error
    
        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."
    
        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"
    
        # If someone accidentally passes x == y, treat it like "plot y"
        if x == y:
            x = None
    
        # Scatter needs x
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."
    
        title_csv = self.csv_name or "current CSV"
    
        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."
    
        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"
    
        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()
        
        return f"Plotted {y} vs {x} as a {plot_type}."

print("Class defined")

csv_backend = CsvManager(RESOURCES_DIR)

node_tools = {
    "list_csv_files": csv_backend.list_csv_files,
    "load_csv": csv_backend.load_csv,
    "get_columns": csv_backend.get_columns,
    "summarize_columns": csv_backend.summarize_columns,
    "describe_column": csv_backend.describe_column,
    "compute_correlation": csv_backend.compute_correlation,
    "plot_data": csv_backend.plot_data,
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources/ folder.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources/ folder and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "CSV filename in resources/, e.g. 'bike_commute.csv'.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the column names of the currently loaded CSV.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Show basic summary statistics for columns (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names. If omitted, summarize all columns.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Show basic summary statistics for a single column (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to describe.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data from the active CSV. If only y is provided, plot y vs row index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {"type": "string", "description": "Column name for y-axis."},
                    "x": {"type": "string", "description": "Optional column name for x-axis."},
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                        "description": "Type of plot to create.",
                    },
                },
                "required": ["y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": (
                "Compute the Pearson correlation coefficient and p-value "
                "between two numeric columns in the currently loaded CSV."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string",
                        "description": "Name of the first numeric column.",
                    },
                    "col2": {
                        "type": "string",
                        "description": "Name of the second numeric column.",
                    },
                },
                "required": ["col1", "col2"],
            },
        },
    },
]

def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run through one react-agent loop using a simple tool-using agent.
    `messages` parameter will usually just contain a system prompt, 
    and then user text will be appended.  

    The loop has three main steps:

    REASON:
      - Call the model with the conversation so far.
      - The model either replies normally, or asks to call a tool from tool set.

    ACT:
      - If tools are requested, run the Python functions

    OBSERVE:
      - Append each requested tool result back into the LLMs conversation history.
      - On the next iteration, the model reads those tool call results and determines
        whether it has reached the goal.

    Stop condition:
      - If the model returns an assistant message with no tool calls, this is the 
        final answer for this react cycle, this implies that reasoning alone without 
        tool calls was enough.  
      - max_tool_rounds is a safety cap to prevent infinite loops.
    """
    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        """
        Return a tool's return value as a message that can be appended to the
        LLMs conversation history. The model will read this tool output on the next
        REASON step.
        """
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        tool_message = {"role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,}
        return tool_message

    for loop_idx in range(max_tool_rounds):
        # REASON: call the model
        # Here it will make use of any previous tool outputs it appended ("observed")
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        # Append the assistant message to the conversation history.
        # Use a plain dict so `messages` stays simple and inspectable.
        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_entry)

        # No tool calls means the model is answering directly.
        if not msg.tool_calls:
            return msg.content 

        # ACT + OBSERVE: run each tool call, then append its result.
        # Note there may be multiple tool calls
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}
                    
            # OBSERVE: append the tool result back into the conversation history.
            messages.append(observe_tool_result(tool_call.id, result))
            
            # After we appending information about all tool outputs, we loop back and REASON again.

    return "I hit the tool-round limit. Try a simpler request."

SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
)

# Q5 -------------------------------------------------------------------------------------------------------
print("\n--- Q5: Correlation With the New Tool ---")

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

result = run_agent_cycle(
    messages,
    "Load bike_commute.csv and compute the correlation between "
    "avg_traffic_density and avg_speed_kmh.",
)

print("\nAgent's final response:")
print(result)

# Q5 observation:
# The agent successfully loaded bike_commute.csv (160 rows, 6 columns)
# and then called compute_correlation with the requested column names.
# Adding the missing tool allowed it to finish without hitting the limit.
#
# Pearson r = -0.5321 indicates a moderate negative linear relationship:
# higher traffic density tends to be associated with lower average speed.
# This association alone does not prove that traffic caused the change.
#
# The reported p-value of 0.0 was rounded to four decimal places by
# our tool; it does not mean the actual probability is exactly zero.
# The model's final answer should have described it as a very small
# p-value rather than presenting the rounded value as exact.

# Q6 -------------------------------------------------------------------------------------------------------
# system: Gives the assistant its instructions and rules for using tools.
# user: Contains the request the agent is trying to complete.
# assistant: Contains the model's tool requests or its final answer.
# tool: Contains the result returned by a Python tool, which the model
#       observes before deciding what to do next.
#
# The history records the tool requests and observations; it does not
# expose the model's private internal reasoning.

print("\n--- Q6: Full Conversation History ---")
print(json.dumps(messages, indent=2, default=str))

# Q6 observation:
# The history contains seven messages: one system message, one user
# request, two assistant tool requests, two tool results, and one
# final assistant answer.
#
# The agent made three completion API calls and executed two local tools.
# It first requested loading, observed the column names, then requested
# the correlation and used that result to produce its final answer.
#
# The assistant messages with content=None contain tool_calls instead
# of a written reply. Each tool result's tool_call_id connects it to
# the matching assistant request.

# --- Lesson 04: Smolagents ----------------------------------------------------------------------

api_key = os.getenv("OPENAI_API_KEY")

# Reusing the CsvManager instance from Lesson 03.
csv_manager = csv_backend


@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.

    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return csv_manager.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.

    Args:
        filename: CSV filename in resources/. You can pass "bike_commute" or "bike_commute.csv".

    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return csv_manager.load_csv(filename)


@tool
def get_columns() -> list[str] | dict:
    """Return column names for the currently loaded CSV.

    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return csv_manager.get_columns()


@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Return summary stats for selected columns (or all columns). 
    This includes count, mean, std, min, max, and percentiles for numeric columns,
    or count, unique, top, freq for categorical columns.

    Args:
        columns: Column names to summarize. If None, summarizes all columns.

    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """
    return csv_manager.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Describe a single column (basic stats) for the requested column.
    This includes count, mean, std, min, max, and percentiles for numeric column,
    or count, unique, top, freq for categorical column.

    Args:
        column: The name of the column to describe.

    Returns:
        A dict of basic stats for the column, or an error dict.
    """
    return csv_manager.describe_column(column)


@tool
def plot_data(y: str, x: str | None = None, plot_type: str = "line") -> str | dict:
    """Plot from the active CSV.

    Args:
        y: Column name to plot on the y-axis. 
        x: Column name to plot on the x-axis. If None, use row index.
        plot_type: "line" or "scatter". Scatter requires x and y.

    Returns:
        Generates and shows the plot. 
        Returns a short success message string, or an error dict/string.
    """
    return csv_manager.plot_data(y=y, x=x, plot_type=plot_type)

# Q7
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute Pearson correlation between two numeric columns in the loaded CSV.

    Args:
        col1: Name of the first numeric column.
        col2: Name of the second numeric column.

    Returns:
        A dictionary containing col1, col2, pearson_r, and p_value,
        or an error dictionary if no CSV is loaded or a column is missing.
    """
    return csv_manager.compute_correlation(col1, col2)


print("\n--- Q7: Automatically Generated Tool Description ---")
print(compute_correlation.description)

# In Q4, I manually wrote the tool's name, description, parameter
# types, and required arguments in a JSON schema.
# Here, @tool builds that metadata from the function name, type hints,
# and docstring. The developer still needs to provide a clear purpose,
# accurate input/output types, and descriptions of every argument.
# It does not automatically understand the function's purpose from its code.


TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,
]

model_to_use = "gpt-4o-mini"  # default model ID
model = OpenAIServerModel(
    api_key=api_key,
    model_id=model_to_use,
)

SYSTEM_PROMPT = (
    "You are a small data assistant to help analyze files stored in resources/. "
    "Use the available tools to do any work requested (do not guess). "
    "Keep answers short and student-friendly."
)

tool_agent = ToolCallingAgent(tools=TOOLS,
                         model=model,
                         instructions=SYSTEM_PROMPT,)

CODE_INSTRUCTIONS = """
You are a helpful CSV analysis assistant.

You can do two kinds of actions:
1) Call the provided tools.
2) Write and execute Python code when tools are not enough.

Rules:
- Prefer tools for simple tasks.
- IMPORTANT: If the user requests plot styling (color, marker, title text, labels, grid, etc.)
  that the plot_data tool cannot control, DO NOT call plot_data.
  Instead, write matplotlib code directly so the plot matches the request.
  If code execution fails, do not fall back to plot_data when the user requested styling (like color). 
  Explain what failed and what you would need to proceed.
- Be honest: only claim you did something if the code or tool actually did it.
- Assume the active dataset lives in csv_manager.df after a CSV is loaded.
"""

code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=CODE_INSTRUCTIONS,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    max_steps=8,
)

# Q8
prompt = (
    "Load bike_commute.csv. Plot avg_heart_rate vs duration_min "
    "as a scatter plot with green dots."
)

print("\n--- Q8: ToolCallingAgent ---")
response_tool = tool_agent.run(prompt)
print("\nToolCallingAgent response:")
print(response_tool)

print("\n--- Q8: CodeAgent ---")
response_code = code_agent.run(
    prompt,
    additional_args={"csv_manager": csv_manager},
)
print("\nCodeAgent response:")
print(response_code)

# Q8 expectations before running:
# The plot_data tool supports x, y, and plot_type, but has no color
# parameter. The ToolCallingAgent can request a scatter plot through
# that tool, but cannot explicitly select green dots.
#
# The CodeAgent can write matplotlib code using csv_manager.df and
# specify color="green", allowing it to handle the extra styling.
#
# Q8 observations:
# The ToolCallingAgent loaded the CSV and called plot_data twice,
# producing two scatter plots with blue dots. Its final response
# incorrectly claimed that the dots were green.
# The tool accepts x, y, and plot_type, but has no color parameter,
# so repeating the same call could not satisfy the styling request.
#
# The CodeAgent loaded the CSV, then wrote matplotlib code using
# csv_manager.df. It explicitly used color="green" and produced
# the requested green scatter plot. It also added a grid and more
# readable axis labels.
#
# A ToolCallingAgent is useful when predefined tools already support
# the requested operations. A CodeAgent can handle additional needs,
# such as custom plot styling, by generating and executing code.
# This run also shows why I should verify actual results instead
# of trusting an agent's claim that it completed every requirement.

# Execution observation:
# Both agents produced their plots and returned final answers.
# However, Matplotlib warned about creating GUI windows outside
# the main thread, and Tkinter reported "main thread is not in
# main loop" errors during cleanup.
# The visible plots demonstrate the agent comparison, but the
# plotting workflow still needs adjustment to finish cleanly.

# Q9
# 1. A ToolCallingAgent would be a good choice for a CSV reporting
# assistant that loads approved datasets, summarizes columns, and
# computes correlations using predefined tools. These tasks have
# clear inputs and established operations, so custom generated code
# is unnecessary. The developer can control the available operations
# through the tools and their validation.
#
# 2. A CodeAgent generates and executes Python code. A mistake in that
# generated code could modify shared data unexpectedly or perform
# unintended operations through the objects and libraries it can access.
# This introduces risks from executing newly generated code, beyond
# the risks of choosing arguments for predefined tools.
# A ToolCallingAgent can still cause harm through powerful tools,
# so its safety also depends on what those tools are allowed to do.