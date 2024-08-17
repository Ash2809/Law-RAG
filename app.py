from flask import Flask, render_template, request
import os
import sys
from dotenv import load_dotenv
from src.workflow import graph_workflow

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ASTRA_API_KEY = os.getenv("ASTRA_API_KEY")
DB_ENDPOINT = os.getenv("DB_ENDPOINT")
DB_ID = os.getenv("DB_ID")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")



app = Flask(__name__)

bot = graph_workflow()

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    inputs = {"question": input}
    for output in bot.stream(inputs):
        for key, value in output.items():
            pass
            # pprint(f"Node '{key}':")
    # pprint("\n---\n")
    response = value["generation"]
    print("Response : ", response)
    return str(response)

if __name__ == '__main__':
    print("Current Working Directory: ", os.getcwd())
    app.run(debug=True)
