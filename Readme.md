# medical-chatbot-stbi
medical chatbot using fine-tuned gemini-1.5-flash, chain-of-thought Retrieval Augmented Generation.
- fine-tuned with a dataset that is partly the same as the med-gemini fine-tuning dataset: https://arxiv.org/abs/2404.18416
- chromadb contains word embeddings of text portions of PubMed abstracts, USMLE prep medical textbooks, statpearls (https://huggingface.co/MedRAG) (https://arxiv.org/abs/2402.13178)


```
- python3 medical_ai_chatbot.py
- tunggu sampai backup chroma db selesai didownload.
- unzip welllahh_chroma.zip
```
