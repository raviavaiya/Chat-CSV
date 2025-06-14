# Langchain Chat-CSV with LLM

This is a Python application that enables you to load a CSV file and ask questions about its contents using natural language. The application leverages Language Models (LLMs) to generate responses based on the CSV data. The LLM will only provide answers related to the information present in the CSV.
<img align="right" alt="Coding" width="500" src="https://github.com/raviavaiya/Chat-CSV/blob/main/csv%20bot.png">

# 🛠️ Tech Stack

- Streamlit
- LangChain
- LLMs
- python-dotenv
- create csv agent

# How it works

The application reads the CSV file and processes the data. It utilizes OpenAI LLMs alongside with Langchain Agents in order to answer your questions. The CSV agent then uses tools to find solutions to your questions and generates an appropriate response with the help of a LLM.

The application employs Streamlit to create the graphical user interface (GUI) and utilizes Langchain to interact with the LLM.

## Installation

To install the repository, follow these steps:

1. Clone this repository to your local machine.
2. Install the necessary dependencies by running the following command:

   ```
   pip install -r requirements.txt
   ```

3. Additionally, you need to obtain an llm API key and add it to the `.env` file.

## Usage

To use the application, execute the `app.py` file using the Streamlit CLI. Make sure you have Streamlit installed before running the application. Run the following command in your terminal:

```
streamlit run app.py
```

