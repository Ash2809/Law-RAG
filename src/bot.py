from flask import Flask, render_template, request
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langgraph.graph import START, StateGraph, END
from typing import Literal, List
from typing_extensions import TypedDict
from dotenv import load_dotenv
import os
import traceback

# Load environment variables
load_dotenv()

# Fetch API keys and endpoints from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ASTRA_API_KEY = os.getenv("ASTRA_API_KEY")
DB_ENDPOINT = os.getenv("DB_ENDPOINT")
DB_ID = os.getenv("DB_ID")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Initialize language model and embeddings
llm = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-1.5-flash", temperature=1)
gemini_embedding = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")

# Initialize vector store
vector_store = AstraDBVectorStore(
    embedding=gemini_embedding,
    api_endpoint=DB_ENDPOINT,
    namespace="constitution",
    token=ASTRA_API_KEY,
    collection_name="Law_bot"
)

# Initialize retriever
retriever = vector_store.as_retriever()

class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str    

def route(state):
    question = state.get("question", "")
    print(f"Routing question: {question}")

    class RouteQuery(BaseModel):
        datasource: Literal["vector-store", "out_of_context"] = Field(..., description="For a Given User Question Find out whether to route it to vector-store or out_of_context")
    
    system = """You are an expert at routing a user question to a vectorstore or out_of_context.
    The vector-store contains documents of The Constitution of India. NOTE: if any question has the words 'according to Indian Constitution' use 'vector-store'.
    Use the vector-store for questions on these topics. Otherwise, use out_of_context."""

    llm_router = llm.with_structured_output(RouteQuery)

    route_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
    question_router = route_prompt | llm_router

    try:
        response = question_router.invoke({"question": question})
        print(f"Routing decision: {response.datasource}")
        if response.datasource == "out_of_context":
            return "out_of_context"
        else:
            return "vector-store"
    except Exception as e:
        print(f"Error in routing: {e}")
        traceback.print_exc()
        return "out_of_context"

def retrieve(state):
    question = state.get("question", "")
    print(f"Retrieving documents for question: {question}")
    try:
        docs = retriever.invoke(question)
        print(f"Retrieved documents: {docs}")
        return {"documents": docs}
    except Exception as e:
        print(f"Error in retrieval: {e}")
        traceback.print_exc()
        return {"documents": []}

def generation(state):
    question = state.get("question", "")
    docs = state.get("documents", [])
    print(f"Generating answer for question: {question}")

    try:
        prompt = hub.pull("rlm/rag-prompt", api_key=LANGCHAIN_API_KEY)
        rag_chain = prompt | llm | StrOutputParser()
        generated_text = rag_chain.invoke({"context": docs, "question": question})
        print(f"Generated text: {generated_text}")
        return {"documents": docs, "question": question, "generation": generated_text}
    except Exception as e:
        print(f"Error in generation: {e}")
        traceback.print_exc()
        return {"documents": docs, "question": question, "generation": ""}

def AnswerGrader(state):
    question = state.get("question", "")
    generation = state.get("generation", "")
    print(f"Grading answer for question: {question}")

    class Grade(BaseModel):
        Binary_Score: str = Field(description="Does the answer resolve the query. yes or no")

    structured_llm = llm.with_structured_output(Grade)

    system = """You are a grader assessing whether an answer addresses / resolves a question \n
    Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    grader_chain = answer_prompt | structured_llm
    try:
        response = grader_chain.invoke({"question": question, "generation": generation})
        print(f"Answer grading: {response.Binary_Score}")
        if response.Binary_Score == "yes":
            return "useful"
        else:
            return "not useful"
    except Exception as e:
        print(f"Error in grading: {e}")
        traceback.print_exc()
        return "not useful"

def Grade_Docs(state):
    question = state.get("question", "")
    documents = state.get("documents", [])
    print(f"Grading documents for question: {question}")

    class Grade_Docs(BaseModel):
        binary_score: Literal["yes", "no"] = Field(..., description="Documents are relevant to the question, 'yes' or 'no'")

    system = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")]
    )

    llm_grader = llm.with_structured_output(Grade_Docs)

    retrieval_grader = grade_prompt | llm_grader

    filtered_docs = []
    for d in documents:
        try:
            score = retrieval_grader.invoke({"question": question, "document": d.page_content})
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
        except Exception as e:
            print(f"Error grading document: {e}")
            traceback.print_exc()

    return {"documents": filtered_docs, "question": question}

def transform_query(state):
    question = state.get("question", "")
    print(f"Transforming question: {question}")

    system = """You are a question re-writer that converts an input question to a better version that is optimized \n
    for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Here is the initial question: \n\n {question} \n Formulate only one improved question."),
        ]
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()
    try:
        revised_question = question_rewriter.invoke({"question": question})
        print(f"Revised question: {revised_question}")
    except Exception as e:
        print(f"Error transforming question: {e}")
        traceback.print_exc()
        revised_question = question  # Fallback

    return {"documents": state.get("documents", []), "question": revised_question}

def decide_to_generate(state):
    filtered_documents = state.get("documents", [])
    if not filtered_documents:
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def out_of_context(state):
    question = state.get("question", "")
    print("The given Question is Out of Context. Please ask relevant questions related to the Indian Constitution.")
