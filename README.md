# Law-RAG
Here's a README file that explains the Flask chatbot project that integrates LangGraph for RAG (Retrieval-Augmented Generation):

---

# Flask Chatbot Project with LangGraph

## Overview

This project is a Flask-based chatbot application that uses advanced language model techniques to handle user queries. The chatbot is designed to route user questions to appropriate data sources, retrieve relevant information, and generate contextually accurate responses. The application leverages **LangGraph** for Retrieval-Augmented Generation (RAG), allowing for enhanced interactions based on the Indian Constitution documents.

### Features

- **Language Model Integration**: Uses Google Generative AI for processing user queries and generating responses.
- **Vector Store**: Utilizes AstraDB as the vector store to store and retrieve relevant documents from the Indian Constitution.
- **Routing and Grading**: The chatbot can route questions to different data sources and assess the relevance of retrieved documents.
- **Query Transformation**: Improves user queries for better retrieval results.
- **Document- Grading**: Detects and Grades the retrived Documents.

## Technologies Used

- **Flask**: A lightweight WSGI web application framework used for building the chatbot's backend.
- **LangGraph**: For creating a state graph that defines the workflow for handling user queries.
- **Google Generative AI**: Used for language model tasks like question routing, document retrieval, and response generation.
- **AstraDB**: A managed database service that stores the vectorized representations of documents.

## Setup and Installation

### Prerequisites

- Python 3.7+
- Pip (Python package installer)
- Flask
- AstraDB account and vector store setup
- Google Generative AI API access

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/flask-chatbot-langgraph.git
   cd flask-chatbot-langgraph
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:

   Create a `.env` file in the project root directory and add the following variables:

   ```bash
   GOOGLE_API_KEY=your_google_api_key
   ASTRA_API_KEY=your_astra_api_key
   DB_ENDPOINT=your_astra_db_endpoint
   DB_ID=your_astra_db_id
   LANGCHAIN_API_KEY=your_langchain_api_key
   ```

4. **Run the Application**:

   ```bash
   python app.py
   ```

## Usage

### Frontend

- The chatbot interface is served by Flask and can be accessed via a web browser. It features a clean, responsive design, allowing users to interact with the bot by typing their questions.

### Backend Workflow

- **LangGraph Integration**:
  - **Routing**: Determines whether the user's query should be routed to a vector store containing Indian Constitution documents or handled as an out-of-context query.
  - **Document Retrieval**: Fetches relevant documents from AstraDB based on the user's query.
  - **Query Transformation**: If the retrieved documents are not relevant, the query is refined for better retrieval.
  - **Response Generation**: Generates a response using the retrieved documents and the Google Generative AI model.
  - **Answer Grading**: Grades the generated response to determine if it resolves the user's query.

### Example Workflow

When a user submits a question, the following steps are executed:

1. **Routing**: The question is analyzed to decide if it should be routed to the vector store or handled as out-of-context.
2. **Document Retrieval**: Relevant documents are retrieved from the AstraDB vector store.
3. **Query Transformation**: If necessary, the query is refined and reprocessed.
4. **Response Generation**: A response is generated based on the retrieved documents.
5. **Answer Grading**: The response is graded to ensure it adequately addresses the query.

If the response is deemed useful, it is returned to the user. Otherwise, the query is transformed, and the process is repeated.

## Contributing

Contributions are welcome! Feel free to submit issues, fork the repository, and make pull requests. Please ensure that your contributions align with the project's objectives and coding standards.
