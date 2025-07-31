from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(
    model="llama3.2:3b",
)

template = """
You are an expert in answeritn questions about a pizza restaurant
Here are some relevant reviews:{reviews}
Here is the question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n---------------------------------")
    question = input("Ask me a question (q to quit) :")
    print("\n\n")
    if question == "q":
        break
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(f"Answer: {result}")
