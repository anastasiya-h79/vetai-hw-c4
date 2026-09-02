#!pip install langgraph langchain langchain-openai --quiet

# import os
# from google.colab import userdata
# os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")

"""
Прототип v2 "Умного ветеринарного помощника" (MVP).
Обновлено по материалам:
  - https://reference.langchain.com/python/langchain/agents/  -> используем create_agent()
    для агентов, вызывающих инструменты в цикле, вместо ручных функций-нод.
  - https://www.pinecone.io/learn/retrieval-augmented-generation/ -> augmented prompt
    и терминология RAG (ingestion/retrieval/augmentation/generation) приведены в соответствие.

Manager (LangGraph StateGraph) координирует агентов, безопасность (Safety Gate) остаётся
жёстким rule-based узлом ВНЕ tool-calling цикла — это осознанное решение (см. vet_design.md).

Запуск в Google Colab:
    !pip install langgraph langchain langchain-openai --quiet
"""

import os
#from typing import TypedDict, Annotated, List, Optional
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# 0. Заглушка векторной базы протоколов (в проде: chunk -> embed -> Qdrant,
#    см. vet_design.md, раздел RAG Flow).
# ---------------------------------------------------------------------------

PROTOCOL_CHUNKS = [
    {
        "disease": "FLUTD",
        "text": "Затруднённое или отсутствующее мочеиспускание у кота — признак "
                "возможной обструкции мочевыводящих путей. Жизнеугрожающее состояние.",
    },
    {
        "disease": "URI",
        "text": "Чихание, выделения из носа, слезотечение без затруднения дыхания — "
                "типичные признаки инфекции верхних дыхательных путей у кошек.",
    },
    {
        "disease": "Dermatitis",
        "text": "Локальное покраснение кожи, расчёсы, выпадение шерсти пятнами — "
                "признаки дерматита, часто блошиного происхождения.",
    },
]

RED_FLAG_KEYWORDS = [
    "не мочится", "не ходит в туалет", "затруднённое дыхание",
    "судороги", "кровотечение", "потеря сознания", "не встаёт",
]


# ---------------------------------------------------------------------------
# 1. Tool для Diagnostic Agent — retrieval-шаг RAG.
#    В create_agent() модель сама решает, когда и с каким запросом его вызвать
#    (agentic RAG, как описано в Pinecone: агент формулирует запрос к retriever,
#    оценивает найденное, при необходимости повторяет с другой формулировкой).
# ---------------------------------------------------------------------------

@tool
def search_veterinary_protocols(query: str) -> str:
    """Ищет релевантные фрагменты ветеринарных протоколов по симптомам.
    В реальной системе: hybrid search (dense+BM25) -> dedupe -> cross-encoder rerank."""
    matches = [c["text"] for c in PROTOCOL_CHUNKS
               if any(w in query.lower() for w in c["text"].lower().split()[:4])]
    return "\n".join(matches) if matches else "Релевантных протоколов не найдено."


def check_red_flags(symptom_text: str) -> Optional[str]:
    """Rule-based Safety-проверка — намеренно НЕ агент и НЕ tool, а жёсткая функция,
    которая не может быть обойдена решением модели (см. vet_design.md, раздел 1)."""
    for kw in RED_FLAG_KEYWORDS:
        if kw in symptom_text.lower():
            return kw
    return None


# ---------------------------------------------------------------------------
# 2. Diagnostic Agent — создан через create_agent(), с augmented-промптом
#    по образцу из Pinecone: "если в CONTEXT нет ответа — так и скажи".
# ---------------------------------------------------------------------------

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # замените на доступную вам модель

diagnostic_agent = create_agent(
    model=model,
    tools=[search_veterinary_protocols],
    system_prompt=(
        "Ты — Diagnostic Agent ветеринарного помощника. "
        "Используй инструмент search_veterinary_protocols, чтобы найти релевантные "
        "протоколы по описанным симптомам. Дай список гипотез с confidence (высокий/средний/низкий). "
        "Если найденный контекст не содержит ответа на вопрос — прямо скажи, что данных недостаточно, "
        "и не придумывай диагноз."
    ),
)

recommendation_agent = create_agent(
    model=model,
    tools=[],  # рекомендации формируются только из переданного контекста, без своих инструментов
    system_prompt=(
        "Ты — Recommendation Agent. Дай базовые рекомендации по уходу СТРОГО на основе "
        "предоставленного контекста (гипотез от Diagnostic Agent). "
        "Обязательно добавь дисклеймер: это не заменяет очный осмотр ветеринара."
    ),
)


# ---------------------------------------------------------------------------
# 3. Состояние графа верхнего уровня (Manager / Supervisor)
# ---------------------------------------------------------------------------

class VetState(TypedDict):
    messages: Annotated[List, operator.add]
    symptom_text: str
    red_flag: Optional[str]
    red_flag_checked: bool
    diagnosis_result: str
    next_step: str


def manager_node(state: VetState) -> VetState:
    if not state.get("red_flag_checked"):
        return {"next_step": "safety_agent"}
    if state.get("red_flag"):
        return {"next_step": "emergency"}
    if not state.get("diagnosis_result"):
        return {"next_step": "diagnostic_agent"}
    return {"next_step": "recommendation_agent"}


def safety_agent_node(state: VetState) -> VetState:
    flag = check_red_flags(state["symptom_text"])
    return {"red_flag": flag, "red_flag_checked": True}


def emergency_node(state: VetState) -> VetState:
    msg = (
        f"⛔ Обнаружены признаки возможного неотложного состояния («{state['red_flag']}»). "
        "Рекомендации по уходу не предоставляются — обратитесь к ветеринару как можно скорее."
    )
    return {"messages": [AIMessage(content=msg)], "next_step": "end"}


def diagnostic_agent_node(state: VetState) -> VetState:
    # create_agent() принимает и возвращает список сообщений — стандартный формат LangChain agents
    result = diagnostic_agent.invoke({
        "messages": [HumanMessage(content=f"Симптомы: {state['symptom_text']}")]
    })
    diagnosis_text = result["messages"][-1].content
    return {"diagnosis_result": diagnosis_text}


def recommendation_agent_node(state: VetState) -> VetState:
    result = recommendation_agent.invoke({
        "messages": [HumanMessage(content=(
            f"Симптомы: {state['symptom_text']}\n"
            f"Гипотезы Diagnostic Agent: {state['diagnosis_result']}"
        ))]
    })
    return {"messages": [AIMessage(content=result["messages"][-1].content)], "next_step": "end"}


# ---------------------------------------------------------------------------
# 4. Сборка графа верхнего уровня (Supervisor + обязательный Safety Gate)
# ---------------------------------------------------------------------------

graph = StateGraph(VetState)
graph.add_node("manager", manager_node)
graph.add_node("safety_agent", safety_agent_node)
graph.add_node("emergency", emergency_node)
graph.add_node("diagnostic_agent", diagnostic_agent_node)
graph.add_node("recommendation_agent", recommendation_agent_node)

graph.set_entry_point("manager")
graph.add_conditional_edges(
    "manager",
    lambda state: state["next_step"],
    {
        "safety_agent": "safety_agent",
        "emergency": "emergency",
        "diagnostic_agent": "diagnostic_agent",
        "recommendation_agent": "recommendation_agent",
        "end": END,
    },
)
graph.add_edge("safety_agent", "manager")
graph.add_edge("diagnostic_agent", "manager")
graph.add_edge("emergency", END)
graph.add_edge("recommendation_agent", END)

app = graph.compile()


# ---------------------------------------------------------------------------
# 5. Примеры запуска (без if __name__ — чтобы гарантированно выполнилось в Colab)
# ---------------------------------------------------------------------------

critical_case = {
    "messages": [HumanMessage(content="Кот третий день не мочится, вялый")],
    "symptom_text": "кот не мочится третий день, вялый",
    "red_flag": None,
    "red_flag_checked": False,
    "diagnosis_result": "",
    "next_step": "",
}
result_1 = app.invoke(critical_case)
print("=== Кейс 1 (ожидаем экстренный ответ) ===")
print(result_1["messages"][-1].content)

mild_case = {
    "messages": [HumanMessage(content="У кота чихание и выделения из носа")],
    "symptom_text": "чихание и выделения из носа второй день",
    "red_flag": None,
    "red_flag_checked": False,
    "diagnosis_result": "",
    "next_step": "",
}
result_2 = app.invoke(mild_case)
print("\n=== Кейс 2 (ожидаем базовые рекомендации) ===")
print(result_2["messages"][-1].content)
