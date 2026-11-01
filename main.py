'''
# pip install langgraph langchain langchain-openai langchain-groq langchain-community langchain-tavily psycopg[binary] psycopg_pool python-dotenv tavily-python pip install requests streamlit

# install PostgresSql and create database
CREATE DATABASE langgraph_memory;  ( or open pgadmin4 and create database there )
'''
# LangGraph Multi-Agent Travel Booking System with Long-Term Memory

# main.py

import os
from typing import TypedDict, Annotated #for state typing
import operator

import psycopg
from langgraph.graph import StateGraph, START, END # for execution graph
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage, #for all types of messages
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq #to interact with Groq LLM

from tools.tavily_tool import tavily_search #self-defined tool to search hotels using Tavily API
from tools.flight_tool import search_flights #self-defined tool to search flights using Aviation API
from dotenv import load_dotenv #to load .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b"
)

# State Definition- shared state/memory across all agents
class TravelState(TypedDict): #for state typing with TypedDict= dictionary with fixed keys and value types
    messages: Annotated[list[AnyMessage], operator.add] #to store all messages exchanged between agents and user, operator.add to append new messages to the list
    #keys for state dictionary with type 
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int

# Flight Agent
def flight_agent(state: TravelState): #node 1: flight_agent input: state(user_query), output: flight_results, messages, llm_calls
    query = state["user_query"]
    flight_data = search_flights(query)
    return {
        "flight_results": flight_data, #update state with flight results
        "messages": [
            AIMessage(content=f"Flight results fetched") #update state with message as all messages should be stored in state["messages"] for long-term memory
        ],
        #update state with llm calls which is not done here "llm_calls": state.get("llm_calls", 0) + 1
    }

# Hotel Agent
def hotel_agent(state: TravelState): #node 2: hotel_agent input: state(user_query), output: hotel_results, messages, llm_calls
    query = f"Best hotels for {state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results, #update state with hotel results
        "messages": [
            AIMessage(content="Hotel information fetched") #update state with message same as node 1
        ],
        #update state with llm calls which is not done here "llm_calls": state.get("llm_calls", 0) + 1
    }

# Itinerary Agent
def itinerary_agent(state: TravelState): #node 3: itinerary_agent input: state(user_query, flight_results, hotel_results), output: itinerary, messages, llm_calls

    prompt = f"""
    Create a travel itinerary.
    User Query:
    {state['user_query']}

    Flight Results:
    {state['flight_results']}

    Hotel Results:
    {state['hotel_results']}
    """

    response = llm.invoke([
        SystemMessage(
            content="You are an expert travel planner" #system prompt
        ),
        HumanMessage(content=prompt) #user prompt
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1 #update state with llm calls which is done here as this agent is calling the LLM
    }

# Final Response Agent
def final_agent(state: TravelState): #node 4: final_agent input: state(user_query, flight_results, hotel_results, itinerary), output: final_response, messages, llm_calls

    final_prompt = f"""
    Generate final travel response.

    Flights:
    {state['flight_results']}

    Hotels:
    {state['hotel_results']}

    Itinerary:
    {state['itinerary']}
    """

    response = llm.invoke([
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# Graph Construction for Multi-Agent System
graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# Persistent connection so both CLI and Streamlit can share the compiled app
_conn = psycopg.connect(DATABASE_URL,autocommit=True)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

app = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "user_utkarsh", #to identify the user and store the state in database for that user
        }
    }

    user_input = input("Enter travel request: ") #take input from user

    result = app.invoke( #invoke the compiled app with the user input and config
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input, #update state with user query
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config #update state with config
    )

    print("\nFINAL RESPONSE:\n")

    for msg in result["messages"]:
        print(msg.content)
