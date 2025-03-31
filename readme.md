# Employee Onboarding Assistant - Saran AI Labs

## 📌 Overview

This project involves building an **onboarding assistant for Saran AI Labs**, an AI research and development company. The assistant will help new employees understand company policies, answer queries about internal regulations, and provide **personalized responses** based on employee details.

The assistant is developed using **Streamlit for UI** and **LangChain with Retrieval Augmented Generation (RAG)** for integrating a vector store containing **company regulations** extracted from a provided PDF.

### 🔍 Data Sources:
1. **Employee-Specific Data**: Synthetic data simulating an employee database.
2. **Company Policies & Regulations**: Extracted from a PDF, stored in a **vector database**, and used to answer company-related queries.

## 🏢 Business Context: Employee Onboarding at Saran AI Labs

Saran AI Labs specializes in **AI model development, deep learning research, and intelligent systems**. To ensure a smooth onboarding experience, the assistant helps employees understand **company rules, benefits, work schedules, safety protocols, and more**.

The chatbot will **personalize its responses** based on employee-specific details and retrieve relevant policy information from a **vector store**.

---

## 🛠 Project Steps

### 1️⃣ Fork & Setup the GitHub Repository
1. Fork the **provided GitHub repository**.
2. Clone it to your local machine:
   ```bash
   git clone <your-repository-url>
   ```
3. Navigate to the project directory:
   ```bash
   cd saran-ai-onboarding-assistant
   ```
4. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### 2️⃣ Display Employee Data
Use **Streamlit’s sidebar** (`st.sidebar`) to display employee details. The **synthetic employee data** includes:
- 🏷️ **Name**
- 🆔 **Employee ID**
- 🏢 **Position & Department**
- 📅 **Date of Hire**
- 📍 **Work Location**
- 🛠 **Skills**

Example Code:
```python
from synthetic_data import generate_employee_data
employee_info = generate_employee_data()
st.sidebar.write(employee_info)
```

---

### 3️⃣ Ingest Company Policies into a Vector Store
Convert the **company policy PDF** into **embeddings** and store it in a **vector store** (FAISS, Pinecone, etc.).

#### Steps:
1. **Split the document** into chunks.
2. **Generate embeddings** for each chunk.
3. **Store embeddings** in a vector database.

Example Code:
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# Load PDF
loader = PyPDFLoader("saran_ai_policies.pdf")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Convert to embeddings
embedding_model = OpenAIEmbeddings()
vector_store = FAISS.from_documents(chunks, embedding_model)
```

---

### 4️⃣ Build the Chatbot Interface
The chatbot will **process user queries** and respond using:
- **Employee-specific data** (e.g., role, department).
- **Company regulations** stored in the vector database.

Example Queries:
- *"What are the safety protocols in the AI research lab?"*
- *"Can you explain the company’s leave policy?"*
- *"What benefits am I eligible for as a new employee?"*

---

### 5️⃣ Structure the Application Using LangChain
Use **LangChain** to manage chatbot interactions.

#### Suggested Code Structure:
- `render()`: Displays the **Streamlit UI**.
- `get_employee_data()`: Fetches synthetic **employee data**.
- `get_response()`: Augments employee data with **retrieved company policies**.
- `retrieve_context()`: Queries the **vector store** using embeddings.

```python
def get_response(employee_info, user_query):
    retrieved_policy = retrieve_context(user_query)
    prompt_template = (
        "You are an onboarding assistant for Saran AI Labs. The employee, {name}, works as a {position} in the {department} department. "
        "Their employee ID is {employee_id}, and their start date was {start_date}. Based on the question: '{user_query}', retrieve any relevant company policy information."
    )
    formatted_prompt = prompt_template.format(**employee_info, user_query=user_query)
    return model.invoke(formatted_prompt)
```

---

### 6️⃣ Implement Streaming Responses
Enable **real-time response streaming** in Streamlit for better user experience.
```python
st.write_stream(response_stream)
```

---

### 7️⃣ Cache Employee Data
Use **Streamlit’s session state** to maintain employee data during the session.
```python
if 'employee_data' not in st.session_state:
    st.session_state['employee_data'] = generate_employee_data()
```

---

### 8️⃣ Test Common Employee Use Cases
Test the chatbot by simulating different employee interactions:

- **Scenario 1:** *"What is the remote work policy?"*
- **Scenario 2:** *"How can I report an HR concern?"*
- **Scenario 3:** *"What are the ethical guidelines for AI research at Saran AI Labs?"*

Ensure the chatbot **retrieves company policies** and **incorporates employee-specific details** into responses.

---

## 🛠 Additional Hints

### ✅ Prompt Design:
Ensure the **LLM prompt** includes **employee details** and **retrieved policy information**.
```python
prompt_template = (
    "You are an onboarding assistant for Saran AI Labs. The employee, {name}, works as a {position} in the {department} department. "
    "Their employee ID is {employee_id}, and their start date was {start_date}. Based on the question: '{employee_query}', retrieve relevant policy information."
)
```

### 🔎 Querying the Vector Store:
Retrieve **relevant information** by converting queries into embeddings.
```python
retrieved_docs = vector_store.similarity_search(query)
```

### 🎨 UI Enhancements:
- 📌 Display **employee details** in the sidebar.
- 💬 Use **Streamlit’s chat UI** for better user interaction.
- 🚀 Ensure **real-time responses** with **streaming output**.

---

## 🚀 Deliverables
✅ **Streamlit app** displaying synthetic employee data.
✅ **Chatbot interface** responding with company policies.
✅ **LangChain integration** for managing prompts & retrievals.
✅ **Real-time streaming** for smoother conversations.
✅ **Session-based caching** for storing employee data.

Once completed, push your changes to **GitHub** and test your chatbot.

🎉 **Good luck & happy coding!** 🎉

