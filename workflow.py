from typing import Annotated, Dict, List, TypedDict
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import Optional
from agents import (
    init_model,
    RouterAgent,
    CVQAAgent,
    MatchingAgent,
    scan_directory,
    extract_text_from_pdf
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    messages: List[BaseMessage]
    next_agent: str
    candidates: List[Dict]
    current_job_description: Optional[str]

def create_router(llm):
    router_agent = RouterAgent(llm)
    
    def route(state: AgentState) -> AgentState:
        next_agent = router_agent.route(state["messages"])
        logger.info(f"\nUser's input: {state['messages'][-1].content}")
        logger.info(f"Router selected agent: {next_agent}\n")
        return state | {"next_agent": next_agent}
    
    return route

def create_cv_qa(llm):
    cv_qa = CVQAAgent(llm)
    
    def answer(state: AgentState) -> AgentState:
        logger.info("CV QA processing...")
        messages = state["messages"]
        candidates = state["candidates"]
        last_message = messages[-1].content
        
        # Answer question about candidates
        response = cv_qa.answer_question(last_message, candidates)
        logger.info("CV QA generated response")
        
        return state | {
            "messages": messages + [AIMessage(content=response)],
        }
    
    return answer

def create_matcher(llm):
    matcher = MatchingAgent(llm)
    
    def match(state: AgentState) -> AgentState:
        logger.info("Matcher processing...")
        messages = state["messages"]
        candidates = state["candidates"]
        last_message = messages[-1].content
        
        # Extract job description from the message if not already set
        if not state["current_job_description"]:
            state = state | {"current_job_description": last_message}
        
        rankings = matcher.rank_candidates(state["current_job_description"], candidates)
        logger.info("Matcher generated rankings")
        
        return state | {
            "messages": messages + [AIMessage(content=rankings)],
        }
    
    return match

def end_conversation(state: AgentState) -> AgentState:
    """End the conversation with a farewell message."""
    logger.info("Ending conversation...")
    messages = state["messages"]
    return state | {
        "messages": messages + [AIMessage(content="Let me know if you need anything else.")]
    }

def create_workflow(use_ollama: bool = True, model_id: str = "phi3:mini"):
    # Initialize the model
    llm = init_model(use_ollama=use_ollama, model_id=model_id)
    logger.info(f"Initializing workflow with model: {model_id}, use_ollama: {use_ollama}")
    
    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", create_router(llm))
    workflow.add_node("cv_qa", create_cv_qa(llm))
    workflow.add_node("matcher", create_matcher(llm))
    workflow.add_node("end", end_conversation)
    
    # Add edges from specialized agents back to router
    workflow.add_edge("cv_qa", "end")
    workflow.add_edge("matcher", "end")
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional edges from router to other agents
    workflow.add_conditional_edges(
        "router",
        lambda x: x["next_agent"],
        {
            "cv_qa": "cv_qa",
            "matcher": "matcher",
            "end": "end"
        }
    )
    
    # Mark end as a terminal node
    workflow.set_finish_point("end")
    
    logger.info("Workflow created successfully")
    return workflow.compile() 