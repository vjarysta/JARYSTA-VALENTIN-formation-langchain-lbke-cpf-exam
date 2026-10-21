"""Chatbot helping the user plan their ideal holidays."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from simple_agent.graph import model

TRIPS = [
    {
        "name": "Randonnée camping en Lozère",
        "labels": ["sport", "mountain", "countryside"],
        "disability_access": False,
    },
    {
        "name": "5 étoiles à Chamonix option fondue",
        "labels": ["mountain", "relaxation"],
        "disability_access": True,
    },
    {
        "name": "5 étoiles à Chamonix option ski",
        "labels": ["mountain", "sport"],
        "disability_access": False,
    },
    {
        "name": "Palavas de paillotes en paillotes",
        "labels": ["beach", "city", "relaxation", "paillote"],
        "disability_access": True,
    },
    {
        "name": "5 étoiles en rase campagne",
        "labels": ["countryside", "relaxation"],
        "disability_access": True,
    },
]


class Criteria(BaseModel):
    """Critères de vacances. True = souhaité, False = refusé, None = pas mentionné."""

    beach: bool | None = Field(default=None, description="Vacances à la plage")
    mountain: bool | None = Field(default=None, description="Vacances à la montagne")
    city: bool | None = Field(default=None, description="Vacances en ville")
    sport: bool | None = Field(default=None, description="Vacances sportives")
    relaxation: bool | None = Field(default=None, description="Vacances détente")
    disability_access: bool | None = Field(
        default=None, description="Besoin d'un accès pour personne en situation de handicap"
    )


class HolidayState(MessagesState):
    criteria: dict[str, bool]


def extract_criteria(state: HolidayState) -> dict:
    criteria = state.get("criteria", {})
    prompt = SystemMessage(
        "Tu extrais les critères de vacances exprimés par l'utilisateur dans son dernier message. "
        "Mets True si l'utilisateur souhaite le critère, False s'il le refuse, "
        "et None s'il n'en parle pas dans son dernier message.\n"
        f"Critères déjà connus : {criteria}"
    )
    new_criteria = model.with_structured_output(Criteria).invoke([prompt, *state["messages"]])
    return {"criteria": {**criteria, **new_criteria.model_dump(exclude_none=True)}}


def route(state: HolidayState) -> Literal["ask_criteria", "suggest_trip"]:
    if state.get("criteria"):
        return "suggest_trip"
    return "ask_criteria"


def ask_criteria(state: HolidayState) -> dict:
    prompt = SystemMessage(
        "Tu es un conseiller en voyages. L'utilisateur n'a encore donné aucun critère. "
        "Si son dernier message n'a pas de sens, dis-lui que tu ne l'as pas compris. "
        "Ensuite, demande-lui ses envies parmi ces critères : plage, montagne, ville, "
        "sport, détente, accès handicapé."
    )
    response = model.invoke([prompt, *state["messages"]])
    return {"messages": [response]}


def filter_trips(criteria: dict[str, bool]) -> list[dict]:
    results = []
    for trip in TRIPS:
        if criteria.get("disability_access") and not trip["disability_access"]:
            continue
        if any(not wanted and name in trip["labels"] for name, wanted in criteria.items()):
            continue
        score = sum(1 for name, wanted in criteria.items() if wanted and name in trip["labels"])
        results.append((score, trip))
    results.sort(key=lambda result: result[0], reverse=True)
    return [trip for _, trip in results]


def suggest_trip(state: HolidayState) -> dict:
    trips = filter_trips(state["criteria"])
    prompt = SystemMessage(
        "Tu es un conseiller en voyages. Propose à l'utilisateur le voyage le plus adapté "
        "parmi la liste ci-dessous (triée du plus pertinent au moins pertinent) "
        "et explique pourquoi en une ou deux phrases. "
        "Si la liste est vide, dis qu'aucun voyage ne correspond et invite-le à changer ses critères. "
        "Tu peux lui demander d'autres critères pour affiner.\n"
        f"Critères de l'utilisateur : {state['criteria']}\n"
        f"Voyages possibles : {trips}"
    )
    response = model.invoke([prompt, *state["messages"]])
    return {"messages": [response]}


builder = StateGraph(HolidayState)
builder.add_node("extract_criteria", extract_criteria)
builder.add_node("ask_criteria", ask_criteria)
builder.add_node("suggest_trip", suggest_trip)

builder.add_edge(START, "extract_criteria")
builder.add_conditional_edges("extract_criteria", route)
builder.add_edge("ask_criteria", END)
builder.add_edge("suggest_trip", END)

graph = builder.compile(name="holiday_agent")
