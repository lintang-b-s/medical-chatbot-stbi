from InstructorEmbedding import INSTRUCTOR

from langchain_community.embeddings import HuggingFaceInstructEmbeddings


from langchain_chroma import Chroma
from dspy.retrieve.chromadb_rm import ChromadbRM
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
from langchain.docstore.document import Document

import sys
import os
import dspy
import google.generativeai as genai

import gdown
import zipfile
#https://drive.google.com/file/d/1S0nxpfC4ifrktLpoetCrNmTyTVSydsqM/view?usp=sharing

file_id = "1S0nxpfC4ifrktLpoetCrNmTyTVSydsqM"
output_path = "welllahh_chroma.zip"

if not os.path.isfile(output_path):
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


class GeminiLM(dspy.LM):
    def __init__(self, model, api_key=None, endpoint=None, **kwargs):
        genai.configure(api_key=os.environ["GEMINI_API_KEY"] or api_key)

        self.endpoint = endpoint
        self.history = []

        super().__init__(model, **kwargs)
        self.model = genai.GenerativeModel(model)

    def __call__(self, prompt=None, messages=None, **kwargs):
        # Custom chat model working for text completion model
        prompt = "\n\n".join([x["content"] for x in messages] + ["BEGIN RESPONSE:"])

        completions = self.model.generate_content(prompt)
        self.history.append({"prompt": prompt, "completions": completions})

        # Must return a list of strings
        return [completions.candidates[0].content.parts[0].text]


vector_store = Chroma(
    collection_name="welllahh_rag_collection_chromadb",
    embedding_function=instructor_embeddings,
    persist_directory="./chroma_langchain_db2",  # Where to save data locally, remove if not necessary
)


## DSPY
ef = embedding_functions.InstructorEmbeddingFunction(
    model_name="hkunlp/instructor-xl", device="cpu"
)

retriever_model = ChromadbRM(
    "welllahh_rag_collection_chromadb",
    "./chroma_langchain_db2",
    embedding_function=ef,
    k=3,
)

llm = GeminiLM(model="tunedModels/gemini-welllahh-zerotemp-lrfv-3536", temperature=0)

# llm = dspy.LM(model='gemini/gemini-1.5-flash', temperature=0)

dspy.settings.configure(lm=llm, rm=retriever_model)


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()

    answer = dspy.OutputField(
        desc="If it is not in context, answer according to your knowledge. Also if the answer is not in the context, add text from the context that is relevant to the query. Explain your answer in detail. don't say 'the text doesn't mention...' in your answer "
    )
    # answer = dspy.OutputField()


class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()


from dsp.utils import deduplicate

qa = dspy.Predict("question: str -> response: str")


class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=5, max_hops=2):
        super().__init__()

        self.generate_query = [
            dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)
        ]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question):
        # question = qa(
        #     question=question
        #     + "; translate the question to english (make sure to only translate the text and do not answer questions in the prompt)"
        # ).response  # setelah di fine-tuning gaperlu translate,  di kasih prompt seperti ini langsung jawab questionnya
        context = []

        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        pred = self.generate_answer(context=context, question=question)
        translate_answer = qa(
            question=pred.answer + "; translate to Indonesian"
        ).response

        # return dspy.Prediction(context=context, answer=pred.answer)
        return dspy.Prediction(context=context, answer=translate_answer)


balen = SimplifiedBaleen()

ans = balen("Apa gejala-gejala diabetes? Bagaimana cara mengobati diabetes?")
print("context: ", ans.context)
print("answer: ", ans.answer)
