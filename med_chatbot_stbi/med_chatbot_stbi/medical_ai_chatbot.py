from InstructorEmbedding import INSTRUCTOR
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_chroma import Chroma
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
from langchain.docstore.document import Document
import sys
import os
import dspy
import google.generativeai as genai
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.load import dumps, loads

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import gdown
import zipfile
# https://drive.google.com/file/d/1S0nxpfC4ifrktLpoetCrNmTyTVSydsqM/view?usp=sharing


file_id = "1S0nxpfC4ifrktLpoetCrNmTyTVSydsqM"
output_path = "welllahh_chroma.zip"

if not os.path.isfile(output_path) and not os.path.isdir("chroma_langchain_db2"):
    gdown.download(
        f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False
    )

    print("Download completed.")
else:
    print("File already exists in the project root.")


if not os.path.isdir("chroma_langchain_db2"):
    print("Extract backup chromadb...")
    with zipfile.ZipFile(output_path, "r") as zip_ref:
        zip_ref.extractall()
    print("Extraction completed.")
    

load_dotenv()


os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")


model_name = "hkunlp/instructor-xl"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)



translate_model = genai.GenerativeModel("gemini-1.0-pro")  # buat translate


def translate_text(text, language):
    text = text + f"; translate to {language}. please just translate the text and don't answer the questions!"
    return translate_model.generate_content(text).candidates[0].content.parts[0].text


summarizer_model = genai.GenerativeModel("gemini-1.5-flash-8b")


def user_summarizer(text):
    return summarizer_model.generate_content(text).candidates[0].content.parts[0].text



retriever_model = Chroma(
    collection_name="welllahh_rag_collection_chromadb",
    persist_directory="./chroma_langchain_db2",
    embedding_function=instructor_embeddings,
)

retriever = retriever_model.as_retriever(search_kwargs={"k": 5})


# template = """You are an AI language model assistant. Your task is to generate search query of the given user question to retrieve relevant documents from a vector database.  Original question: {question}""" #gabisa malah gak bikin query
template = """Write a simple search queries that will help answer complex user questions, make sure in your answer you only give one simple query and don't include any other text! . Original question: {question}"""
search_query_prompt = ChatPromptTemplate.from_template(template)


class GeminiLLM(LLM):
    """A custom chat model.

    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying models documentation or API.

    Example:

        .. code-block:: python

            model = CustomChatModel(n=2)
            result = model.invoke([HumanMessage(content="hello")])
            result = model.batch([[HumanMessage(content="hello")],
                                 [HumanMessage(content="world")]])
    """

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT include the prompt.
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        # return prompt[: self.n]
        llm = genai.GenerativeModel(
            model_name="tunedModels/gemini-welllahh-zerotemp-lrfv-3536"  # sebelumnya 0
        )  # buat jawab pertanyaan medis

        ans = llm.generate_content(prompt, generation_config={'temperature': 0.05}).candidates[0].content.parts[0].text
        return ans

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "gemini-welllahh-zerotemp-lrfv-3536"


llm = GeminiLLM()

generate_query = search_query_prompt | llm | StrOutputParser()


def add_websearch_results(query):
    results = DDGS().text(query, max_results=8)

    websearch = []

    for res in results:
        if "webmd" in res["href"]:
            continue
        if len(websearch) == 3:
            break
        if ".org" in res["href"] or ".gov" in res["href"]:

            link = res["href"]
            page = requests.get(link).text
            doc = BeautifulSoup(page, "html.parser")
            text = ""
            hs = doc.find_all("h2")
            h3s = doc.find_all("h3")
            for h3 in h3s:
                hs.append(h3)

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
                        for li in adjacent.find_all("li"):
                            text += ": " + li.text + ","
                        text += "\n"
            if "Why have I been blocked" in text or text == "":
                continue

            websearch.append(text)
    return websearch


class DuckDuckGoRetriever(BaseRetriever):
    """A toy retriever that contains the top k documents that contain the user query.

    This retriever only implements the sync method _get_relevant_documents.

    If the retriever were to involve file access or network access, it could benefit
    from a native async implementation of `_aget_relevant_documents`.

    As usual, with Runnables, there's a default async implementation that's provided
    that delegates to the sync implementation running on another thread.
    """

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


websearch_retriever = DuckDuckGoRetriever(k=3)



def add_web_search(query):
    websearch = add_websearch_results(query)


retrieval_chain = (
    generate_query
    | {"chroma": retriever, "websearch": websearch_retriever}
)


def format_docs(docs):
    chroma_docs = [doc.metadata["content"] + doc.page_content for doc in docs["chroma"]]
    docs = list(set(chroma_docs + docs["websearch"]))
    context = "\n\n".join(doc for doc in docs)
    return context

# add a summary of the text in context to your answer
# if the answer to the user's question is not in the context and you can't answer user questions and don't repeat your answers, add text from context that is relevant to the user's question . Don't say 'the given text does not answer the user's question or is not relevant to the user's question' in your answer. 
# please don't repeat the same answer!
# https://arxiv.org/pdf/2205.11916 ,https://arxiv.org/pdf/2005.11401
template = """Answer the question based only on the following context:
{context}
\n

please do not mention the same answer more than once and Don't say 'the given text does not answer the user's question or is not relevant to the user's question' in your answer. just answer the question don't add any other irrelevant text
\n
Question: {question}
\n
Answer: Let's think step by step in order to
"""


prompt = ChatPromptTemplate.from_template(template)


answer_chain = (
    {"context": retrieval_chain | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | { "llm_output":  StrOutputParser(), "context": retrieval_chain | format_docs} 
)


def answer(question):

    answer = answer_chain.invoke(input=question)
    return answer


def answer_pipeline(question, chat_history):
    user_context = ""
    if chat_history != "" and "insufficient data" not in chat_history.lower() :
        user_context = user_summarizer(
            text=chat_history
            + "\n"
            + "summarize the user's health condition based on the user's chat history above! only explain the user's health condition and nothing else!"
        )

   
    if user_context != "":
        question = "my health condition: " + user_context + "\n" + question
    question = translate_text(question, "English")
    question = question.replace("\n", "  ")
    print("retrieving relevant passages and answering user question....")
    pred = answer(question)
    translate_answer = translate_text(pred["llm_output"], "Indonesian")
    return translate_answer, pred["context"]
