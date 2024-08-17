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
from pprint import pprint
from src.bot import GraphState, route, retrieve, generation, AnswerGrader, Grade_Docs, transform_query, decide_to_generate, out_of_context




def graph_workflow():
    workflow = StateGraph(GraphState)

    workflow.add_node("out_of_context_node", out_of_context)
    workflow.add_node("retrieve_node", retrieve)
    workflow.add_node("grade_documents_node", Grade_Docs)
    workflow.add_node("generation_node", generation)
    workflow.add_node("transform_query_node", transform_query)

    workflow.add_conditional_edges(START, route, {
        "vector-store": "retrieve_node",
        "out_of_context": "out_of_context_node"
    })

    workflow.add_edge("out_of_context_node", END)
    workflow.add_edge("retrieve_node", "grade_documents_node")

    workflow.add_conditional_edges("grade_documents_node", decide_to_generate, {
        "transform_query": "transform_query_node",
        "generate": "generation_node"
    })

    workflow.add_edge("transform_query_node", "retrieve_node")

    workflow.add_conditional_edges("generation_node", AnswerGrader, {
        "useful": END,
        "not useful": "transform_query_node"
    })

    # Compile workflow
    app = workflow.compile()
    return app

if __name__ == "__main__" :
    app = graph_workflow()
    inputs = {"question": "What is right to speech in indian consitution?"}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Node '{key}':")
    pprint("\n---\n")

# Final generation
    pprint(value["generation"])