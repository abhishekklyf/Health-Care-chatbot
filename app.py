import gradio as gr
from langchain import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant

# Initialize LLM and other components
local_llm = "BioMistral-7B.Q4_K_M.gguf"
llm = LlamaCpp(model_path=local_llm, temperature=0.3, max_tokens=2048, top_p=1, n_ctx=2048)

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Chat History: {chat_history}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
url = "http://localhost:6333"
client = QdrantClient(url=url, prefer_grpc=False)
db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")
retriever = db.as_retriever(search_kwargs={"k": 1})

chat_history = []

# Create the conversational chain
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

def predict(message, history):
    history_langchain_format = []
    response = chain({"question": message, "chat_history": chat_history})
    answer = response['answer']
    chat_history.append((message, answer))
    
    temp = []
    for input_question, bot_answer in history:
        temp.append(input_question)
        temp.append(bot_answer)
        history_langchain_format.append(temp)
    
    temp.clear()
    temp.append(message)
    temp.append(answer)
    history_langchain_format.append(temp)
    
    return answer

# Define Gradio interface
def gradio_predict(user_input, history=[]):
    response = predict(user_input, history)
    history.append((user_input, response))
    return history, history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def respond(message, history):
        history, _ = gradio_predict(message, history)
        return history, ""

    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
