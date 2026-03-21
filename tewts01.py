from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class RAGState(TypedDict):
    question: str
    rewritten_query: str
    docs: list[str]
    answer: str


def rewrite_query_node(state: RAGState):
    question = state["question"]
    rewritten_query = f"请检索与这个问题最相关的资料：{question}"
    return {"rewritten_query": rewritten_query}


def retrieve_node(state: RAGState):
    query = state["rewritten_query"]
    docs = [f"文档A: {query}", f"文档B: {query}"]
    return {"docs": docs}


def generate_answer_node(state: RAGState):
    question = state["question"]
    docs = state["docs"]
    answer = f"问题：{question}\n参考资料：{docs}\n回答：这是基于检索结果生成的答案。"
    return {"answer": answer}


graph = StateGraph(RAGState)

graph.add_node("rewrite", rewrite_query_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_answer_node)

graph.add_edge(START, "rewrite")
graph.add_edge("rewrite", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()

result = app.invoke({
    "question": "DeepJSCC是什么？",
    "rewritten_query": "",
    "docs": [],
    "answer": ""
})

print(result["answer"])