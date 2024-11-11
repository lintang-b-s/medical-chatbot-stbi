from InstructorEmbedding import INSTRUCTOR
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_chroma import Chroma
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
from langchain.docstore.document import Document
import os
import google.generativeai as genai
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.load import dumps, loads

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch
import faiss
import numpy as np
from datasets import load_dataset

ds = load_dataset("HPAI-BSC/medqa-cot")

medqa_cot_embeds = np.load("medqa_cot.npy")

index = faiss.IndexFlatIP(768)
index.add(medqa_cot_embeds)

query_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
query_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")


load_dotenv()


os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")


model_name = "hkunlp/instructor-xl"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
model = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder")


translate_model = genai.GenerativeModel("gemini-1.0-pro")  # buat translate


def translate_text(text, language):
    text = (
        text
        + f"; translate to {language}. please just translate the text and don't answer the questions!"
    )
    return translate_model.generate_content(text).candidates[0].content.parts[0].text


summarizer_model = genai.GenerativeModel("gemini-1.5-flash-8b")


def user_summarizer(text):
    return summarizer_model.generate_content(text).candidates[0].content.parts[0].text


retriever_model = Chroma(
    collection_name="welllahh_rag_collection_chromadb",
    persist_directory="./chroma_langchain_db2",
    embedding_function=instructor_embeddings,
)

retriever = retriever_model.as_retriever(search_kwargs={"k": 8})


# template = """You are an AI language model assistant. Your task is to generate search query of the given user question to retrieve relevant documents from a vector database.  Original question: {question}""" #gabisa malah gak bikin query
template = """Write a simple search queries that will help answer complex user questions, make sure in your answer you only give one simple query and don't include any other text! . Original question: {question}"""
search_query_prompt = ChatPromptTemplate.from_template(template)


class GeminiLLM(LLM):
    """custom model pakai gemini"""

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        llm = genai.GenerativeModel(
            model_name="tunedModels/gemini-welllahh-zerotemp-lrfv-3536"  # sebelumnya 0
        )  # buat jawab pertanyaan medis

        ans = (
            llm.generate_content(prompt, generation_config={"temperature": 0.05})
            .candidates[0]
            .content.parts[0]
            .text
        )
        return ans

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "gemini-welllahh-zerotemp-lrfv-3536"


llm = GeminiLLM()

generate_query = search_query_prompt | llm | StrOutputParser()

cant_access = {
    "npin.cdc.gov",
    "www.ncbi.nlm.nih.gov",
}  # gak bisa diakses & gak muncul tag <p> nya


def add_websearch_results(query):
    results = DDGS().text(query, max_results=7)

    websearch = []

    for res in results:
        domain = res["href"].split("/")[2]
        if "webmd" in res["href"] or ".pdf" in res["href"] or domain in cant_access:
            continue
        if len(websearch) == 4:
            break
        if ".org" in res["href"] or ".gov" in res["href"] or "who" in res["href"]:

            link = res["href"]
            try:
                page = requests.get(link).text
            except requests.exceptions.RequestException as errh:
                print(f"error: {errh}")
                continue
            doc = BeautifulSoup(page, features="html.parser")
            text = ""
            hs = doc.find_all("h2")
            h3s = doc.find_all("h3")
            ps = doc.find_all("p")
            for h3 in h3s:
                hs.append(h3)
            for pp in ps:
                hs.append(pp)

            hs_parents = set()
            for h2 in hs:
                h2_parent = h2.parent
                if h2_parent in hs_parents:
                    continue
                hs_parents.add(h2_parent)
                h2_adjacent = h2_parent.children
                for adjacent in h2_adjacent:
                    if adjacent.name == "p" and adjacent.text != "\n":
                        text += adjacent.text + "\n"
                    if (
                        adjacent.name == "h2"
                        or adjacent.name == "h3"
                        or adjacent.name == "h4"
                    ):
                        text += adjacent.text + ": \n"
                    if adjacent.name == "ul" or adjacent.name == "ol":
                        text += ": "
                        for li in adjacent.find_all("li"):
                            text += li.text + ","
                        text += "\n"
            if "Why have I been blocked" in text or text == "" or text == ": \n":
                continue

            websearch.append(text)
    return websearch


class DuckDuckGoRetriever(BaseRetriever):
    """List of documents to retrieve from."""

    k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Sync implementations for retriever."""
        matching_documents = []
        websearch = add_websearch_results(query)

        for document in websearch:
            if len(matching_documents) > self.k:
                return matching_documents

            matching_documents.append(document)
        return matching_documents


medqa_cot_data = ds["train"]


class MedQACoTRetriever(BaseRetriever):
    """List of documents to retrieve from."""

    k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Sync implementations for retriever."""
        matching_documents = []
        # websearch = add_websearch_results(query)
        inds = []
        with torch.no_grad():
            # tokenize the queries
            encoded = query_tokenizer(
                [query],
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            )
            embeds = query_model(**encoded).last_hidden_state[:, 0, :]
            scores, inds = index.search(embeds, k=self.k)

        relevant_cot = []
        for score, ind in zip(scores[0], inds[0]):
            curr_cot = medqa_cot_data[int(ind)]
            curr_question = curr_cot["question"]
            curr_answer = curr_cot["response"]
            relevant_cot.append(
                f"Question: {curr_question}\nAnswer: Let's think step by step. {curr_answer}"
            )
        return relevant_cot


websearch_retriever = DuckDuckGoRetriever(k=4)
medqa_cot_retriever = MedQACoTRetriever(k=3)


def rerank_docs_medcpt(query, docs):
    pairs = [[query, article] for article in docs]
    with torch.no_grad():
        encoded = tokenizer(
            pairs,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=512,
        )

        logits = model(**encoded).logits.squeeze(dim=1)
        values, indices = torch.sort(logits, descending=True)
        relevant = [docs[i] for i in indices[:6]]
    return relevant


retrieval_chain = generate_query | {
    "chroma": retriever,
    "websearch": websearch_retriever,
    "medqa_cot": medqa_cot_retriever,
    "query": StrOutputParser(),
}


def format_docs(docs):
    query = docs["query"]
    chroma_docs = [doc.metadata["content"] + doc.page_content for doc in docs["chroma"]]
    relevant_cot = docs["medqa_cot"]

    docs = list(set(chroma_docs + docs["websearch"]))
    # rerank passage2 dari document chromadb & hasil scraping webpage hasil duckduckgosearch
    relevant_docs = rerank_docs_medcpt(query, docs)
    context = "\n\n".join(doc for doc in relevant_docs)

    # tambahin few-shot chain-of-thought prompting
    for cot in relevant_cot:
        context += "\n" + cot
    return context


template = """Answer the question based only on the following context:
{context}
\n

please do not mention the same answer more than once and Don't say 'the given text does not answer the user's question or is not relevant to the user's question' in your answer. just answer the question don't add any other irrelevant text
\n
Question: {question}
\n
Answer: Let's think step by step.
"""

prompt = ChatPromptTemplate.from_template(template)


answer_chain = (
    {"context": retrieval_chain | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | {"llm_output": StrOutputParser(), "context": retrieval_chain | format_docs}
)


def answer(question):
    answer = answer_chain.invoke(input=question)
    return answer



def answer_pipeline_for_eval(question):
    """
    buat evaluasi llm.
    Pertanyaan dalam bahasa inggris.
    pakai pertanyaan dari PubmedQA, MedQA -> pakai  (exact match)
    pakai: https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options/viewer/default/test?row=23
    
    """
    question = question.replace("\n", "  ")
    print("retrieving relevant passages and answering user question....")
    pred = answer(question)
    return pred["llm_output"], pred["context"]