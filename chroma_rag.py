import chromadb
import pdfplumber
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
#phi path = r'C:\Users\tybin\OneDrive\Desktop\legalintelligence-1\src\models\phi-2'

class RAG:
    def __init__(self):
        self.client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(base_dir, "..", "db", "vectorstore")
        os.makedirs(db_path, exist_ok=True)

        self.chroma = chromadb.PersistentClient(path=r"db\vectorstore")
        self.collection = self.chroma.get_or_create_collection("docs")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.messages = [
            {
                'role':'system', 
                'content':'''You are a helpful assistant that answers questions based on provided documents.
            Use only the information in the context below to answer.
            If the answer is not found, say you don’t know.'''
            }
        ]

    def pdf_reader(self, file):
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if page_text:
                    text += page_text + "\n"
        return text


    def adding_chunked(self, full_text: str, doc_name:str, topic: str, section: str= "", subsection: str = ""):

        if type(subsection) == list:
            subsection = subsection[0]

        chunks = self.splitter.split_text(full_text)  
        existing_ids = set(self.collection.get()['ids'])

        for i, chunk in enumerate(chunks):
            chunk_id = f"{topic}_{doc_name}_chunk_{i}"

            if chunk_id in existing_ids:
                continue
            
            embedding = self.embedding_model.encode(chunk)


            self.collection.add(
                documents=[chunk],
                embeddings=[embedding.tolist()],
                ids=[chunk_id],
                metadatas=[{
                    "document": doc_name,
                    "topic":topic,
                    "section": section,
                    "subsection":subsection,
                    "chunk_index": i
                    }]
                    )
            
    def query_context(self, query:str, topic: str = "", section: str = "", subsection: str = "", document: str= ""):
        filters = {
            "topic":topic,
            "section":section,
            "subsection":subsection,
            "document":document
        }
        filter_df = {k: v for k, v in filters.items() if v}

        embedding = self.embedding_model.encode(query)

        results_lst = []
        for k, v in filters.items():
            results = self.collection.query(
                query_embeddings=[embedding.tolist()], 
                n_results=3,
                where={k: v}
                )
            results_lst.append(results)
        
        all_docs = []
        for res in results_lst:
            all_docs.extend(res["documents"][0])

        if not all_docs:
            return "No relevant context found."


        context = "\n".join(all_docs)
        return context

    def rag_format(self, query:str, context:str):
        prompt = f"""
            RAG Returned Context:
            {context}

            Question:
            {query}
        """
        return prompt

    def generation(self, formated_prompt):
        # adds conversation to the model
        self.messages.append(
            {
                'role':'user', 
                'content':f'{formated_prompt}'
            }
            )
        
        response = self.client.chat_completion(self.messages, max_tokens=100)
        # extracts txt from the model
        respomse_txt = response.choices[0].message.content

        

        # add assistant response to the model
        self.messages.append({'role':'assistant', 'content':f'{respomse_txt}'})
        
        return respomse_txt
        # response = self.client.text_generation(formated_prompt, max_new_tokens=300)
        # return response


    def ask(self, query, topic="", section="", subsection="", document=""):
        context = self.query_context(query, topic, section, subsection, document)
        formatted = self.rag_format(query, context)
        return self.generation(formatted)

    
