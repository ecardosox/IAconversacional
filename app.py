from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    api_key=api_key
)

user = os.getenv("DB_USER", "postgres")
password = os.getenv("DB_PASSWORD", "postgres")
host = os.getenv("DB_HOST", "localhost")
port = os.getenv("DB_PORT", "5432")
database = os.getenv("DB_NAME", "seguradora")

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    template = """
    Você é um analista de dados de uma empresa. Você está interagindo com um usuário que está fazendo perguntas sobre o banco de dados da empresa.
    Com base no esquema da tabela abaixo, escreva uma consulta SQL que responderia à pergunta do usuário. Considere o histórico da conversa.

    <SCHEMA>{schema}</SCHEMA>

    Histórico da Conversa: {chat_history}

    Escreva apenas a consulta SQL, e nada mais. Não envolva a consulta SQL em nenhum outro texto, nem mesmo com crase.

    Por exemplo:
    Pergunta: Quais os 3 artistas com mais faixas?
    Consulta SQL: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Pergunta: Nomeie 10 artistas
    Consulta SQL: SELECT Name FROM Artist LIMIT 10;

    Sua vez:

    Pergunta: {question}
    Consulta SQL:
    """

    prompt = ChatPromptTemplate.from_template(template)

    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)

    template = """
    Você é um analista de dados de uma empresa. Você está interagindo com um usuário que está fazendo perguntas sobre o banco de dados da empresa.
    Com base no esquema da tabela abaixo, pergunta, consulta SQL e resposta SQL, escreva uma resposta em linguagem natural.

    <SCHEMA>{schema}</SCHEMA>

    Histórico da Conversa: {chat_history}
    Consulta SQL: <SQL>{query}</SQL>
    Pergunta do usuário: {question}
    Resposta SQL: {response}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá! Sou um assistente SQL. Pergunte-me qualquer coisa sobre seu banco de dados."),
    ]

st.set_page_config(page_title="Chat com PostgreSQL", page_icon=":speech_balloon:")

st.title("Chat com PostgreSQL")

with st.sidebar:
    st.subheader("Configurações")
    st.write("Esta é uma aplicação de chat simples usando PostgreSQL. Conecte-se ao banco de dados e comece a conversar.")

    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Porta", value="5432", key="Port")
    st.text_input("Usuário", value="postgres", key="User")
    st.text_input("Senha", type="password", value="postgres", key="Password")
    st.text_input("Banco de Dados", value="seguradora", key="Database")

    if st.button("Conectar"):
        with st.spinner("Conectando ao banco de dados..."):
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Conectado ao banco de dados!")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Digite uma mensagem...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)

    st.session_state.chat_history.append(AIMessage(content=response))