from PyPDF2 import PdfReader
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_groq import ChatGroq 
import pickle
import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
st.header("Chat with CSV" )
llm=ChatGroq(model="Gemma2-9b-It")

# Sidebar contents
with st.sidebar:
    st.title("🤖 CSV Chat App")
    st.markdown('''
    ## About
    This app is an LLM-Powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Ollama](https://ollama.com/) LLM Model
                            
    ''')
    add_vertical_space(5)
    st.write("Made with ❤️ by [Ravi Avaiya](https://raviavaiya-portfolio.vercel.app/)")
    if st.button("🗑️ Clear Chat History"):
        # clear_chat_history()
        st.success("Chat history cleared.")

def main():

    
    user_csv = st.file_uploader("Upload CSV", type="csv")

    if user_csv is not None:
        user_question = st.text_input("Ask something about your CSV:")
        
        
        agent = create_csv_agent(llm, user_csv, verbose=True, allow_dangerous_code=True)


        if user_question is not None and user_question != "":
            response = agent.invoke(user_question)
            st.write(f": {user_question}")
            st.write(f"Answer: {response['output']}")


if __name__ == "__main__":
    main()