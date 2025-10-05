from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
from pydantic import BaseModel, Field
import io
from dotenv import load_dotenv
import os

# LangChain Imports
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --------------------- ENV ---------------------
load_dotenv()

# --------------------- Model for Image Classification ---------------------
model = tf.keras.models.load_model("model/potatoes.h5")
CLASS_NAMES = ["Early_Blight", "Late_Blight", "Healthy"]

# --------------------- FastAPI App ---------------------
app = FastAPI(title="Potato Disease Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

@app.get("/")
def read_index():
    return FileResponse("templates/index.html")

# --------------------- Image Preprocessing ---------------------
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# --------------------- Prediction Endpoint ---------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = preprocess_image(contents)
    preds = model.predict(img)
    pred_class = CLASS_NAMES[np.argmax(preds[0])]
    confidence = float(np.max(preds[0]))
    
    return JSONResponse({
        "class": pred_class,
        "confidence": round(confidence, 4)
    })

# --------------------- LangChain + FAISS Setup ---------------------

# 1. Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 2. Knowledge Base load ya build
loader = TextLoader('knowledge_base.txt')
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 150,
    chunk_overlap = 15
    )
docs = splitter.split_documents(documents)
vectorstore = FAISS.from_texts(docs, embeddings)
vectorstore.save_local("faiss_index")

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 3. HuggingFace Model (LLM)
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-alpha",  
    temperature=0.3,
    max_new_tokens=200
)

# --------------------- Prompt Template ---------------------
prompt = PromptTemplate.from_template(
    """
    آپ ایک زرعی ماہر ہیں۔
    آپ کو context دیا گیا ہے اور ساتھ میں user کا سوال ہے۔
    آپ نے ہمیشہ اردو زبان میں جواب دینا ہے۔
    
    Context:
    {context}
    
    سوال:
    {query}
    
    جواب:
    """
)

# parser
parser = StrOutputParser()

# --------------------- Chat Endpoint ---------------------
class Query(BaseModel):
    q: str = Field("user query")

@app.post("/chat")
async def chat(query: Query):
    # Retrieve docs
    retrieved_docs = retriever.invoke(query.q)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Run chain manually
    chain = prompt | llm | parser
    result = chain.invoke({"query": query.q, "context": context})

    return JSONResponse({
        "query": query.q,
        "answer": result
    })

# --------------------- Run App ---------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
