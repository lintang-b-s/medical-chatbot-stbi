##### from InstructorEmbedding import INSTRUCTOR
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
from concurrent.futures import ThreadPoolExecutor


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
# https://arxiv.org/abs/2212.09741
instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="Represent the Medicine sentence for retrieving relevant documents: ",
)

instructor_embeddings.query_instruction = (
    "Represent the Medicine sentence for retrieving relevant documents: "
)
tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
model = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder")


translate_model = genai.GenerativeModel("gemini-1.0-pro")  # buat translate


def translate_text(text, language):
    newText = text
    if language == "Indonesian":
        newText = text.replace("\n", " \n ")
    newText = (
        newText
        + f"; translate to {language}. please just translate the text and don't answer the questions!"
    )
    return translate_model.generate_content(newText).candidates[0].content.parts[0].text


summarizer_model = genai.GenerativeModel("gemini-1.5-flash-8b")


def user_summarizer(text):
    return summarizer_model.generate_content(text).candidates[0].content.parts[0].text


retriever_model = Chroma(
    collection_name="welllahh_rag_collection_chromadb",
    persist_directory="./chroma_langchain_db2",
    embedding_function=instructor_embeddings,
)

retriever = retriever_model.as_retriever(search_kwargs={"k": 5})


template = """Write multiple different very short search queries (each queries by a separated by "," & maximum different 5 very short search queries) that will help answer complex user questions, make sure in your answer you only give multiple different very short search queries (each queries by a separated by "," & maximum different 5 very short search queries) and don't include any other text! . Original question: {question}"""
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
            llm.generate_content(prompt, generation_config={"temperature": 0.12})
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


def scrape_websearch(queryProc):
    websearch = []
    try:
        results = DDGS().text(queryProc, max_results=8)
    except Exception:
        return websearch
    for res in results:
        domain = res["href"].split("/")[2]
        if "webmd" in res["href"] or ".pdf" in res["href"] or domain in cant_access:
            continue
        if len(websearch) == 3:
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


def add_websearch_results(query):
    queries = [query]
    if "," in query:
        queries.extend(query.split(","))
    else:
        queries = [query]
    websearch_all = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(scrape_websearch, queries)
    for new_contexts in results:
        websearch_all.extend(new_contexts)

    return websearch_all


class DuckDuckGoRetriever(BaseRetriever):
    """ """

    """List of documents to retrieve from."""
    k: int
    """Number of top results to return"""

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

        queries = [query]
        if "," in query:
            queries.extend(query.split(","))
        else:
            queries = [query]

        relevant_cot = []

        def retrieve_cot(search_query):
            with torch.no_grad():
                # tokenize the queries
                encoded = query_tokenizer(
                    [search_query],
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=512,
                )
                embeds = query_model(**encoded).last_hidden_state[:, 0, :]
                scores, inds = index.search(embeds, k=self.k)

            curr_relevant_cot = []
            for score, ind in zip(scores[0], inds[0]):
                curr_cot = medqa_cot_data[int(ind)]
                curr_question = curr_cot["question"]
                curr_answer = curr_cot["response"]
                curr_relevant_cot.append(
                    f"\nQuestion: {curr_question}\nAnswer: Let's think step by step. {curr_answer}"
                )
            return curr_relevant_cot

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = executor.map(retrieve_cot, queries)
        for new_contexts in results:
            relevant_cot.extend(new_contexts)

        return relevant_cot


websearch_retriever = DuckDuckGoRetriever(k=2)
medqa_cot_retriever = MedQACoTRetriever(k=1)


def rerank_docs_medcpt(queries, docs):
    queries = [queries]
    queries.extend(queries[0].split(","))
    relevant = set()

    def process_query(query):
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
            curr_relevant = [docs[i] for i in indices[:3]]
        return curr_relevant

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(process_query, queries)

    for curr_relevant in results:
        relevant.update(curr_relevant)

    relevant = list(relevant)
    return relevant


retrieval_chain = generate_query | {
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
    context = " \n\n".join(doc for doc in relevant_docs)

    # tambahin few-shot chain-of-thought prompting
    relevant_cot = list(set(relevant_cot))[:2]
    for cot in relevant_cot:
        context += " \n" + cot
    return context, relevant_docs


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

answer_chain = prompt | llm | {"llm_output": StrOutputParser()}


def retrieve_and_append(query):
    new_knowledge_base_contexts = retriever.invoke(query)
    return new_knowledge_base_contexts


def answer(question):
    docs = retrieval_chain.invoke(question)
    docs["chroma"] = []
    search_query = [docs["query"]]
    search_query.extend(docs["query"].split(","))

    with ThreadPoolExecutor(max_workers=1) as executor:
        results = executor.map(retrieve_and_append, search_query)
    for new_contexts in results:
        docs["chroma"].extend(new_contexts)

    context, relevant_docs = format_docs(docs)
    answer = answer_chain.invoke({"context": context, "question": question})

    context = f"search query: {','.join(search_query)}\n" + context
    return answer, context, [context] + relevant_docs


def answer_pipeline_for_eval(question):
    """
    buat evaluasi llm.
    Pertanyaan dalam bahasa inggris.
    pakai pertanyaan dari PubmedQA, MedQA -> pakai  (exact match)
    pakai: https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options/viewer/default/test?row=23

    """
    question = question.replace("\n", "  ")
    print("retrieving relevant passages and answering user question....")
    pred, context, relevant_docs = answer(question)
    return pred["llm_output"], context, relevant_docs
