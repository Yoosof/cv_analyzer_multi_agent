import streamlit as st
from copy import deepcopy
from agents import (
    init_model,
    DocumentClassifierAgent,
    CVAnalyzerAgent,
    scan_directory,
    extract_text_from_pdf,
    HUGGINGFACE_AVAILABLE
)
from workflow import create_workflow
from langchain_core.messages import HumanMessage, AIMessage
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "candidates" not in st.session_state:
        st.session_state.candidates = []
    if "workflow" not in st.session_state:
        st.session_state.workflow = None
    if "cv_loaded" not in st.session_state:
        st.session_state.cv_loaded = False
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "backend" not in st.session_state:
        st.session_state.backend = "ollama"

def load_cvs(directory_path: str, backend: str, model_id: str, api_key: str = None):
    logger.info(f"Loading CVs from directory: {directory_path}")
    # Initialize model for CV classification and analysis
    llm = init_model(model_id=model_id, backend=backend, api_key=api_key)
    classifier = DocumentClassifierAgent(llm)
    cv_analyzer = CVAnalyzerAgent(llm)
    
    # Scan directory for PDFs
    pdf_files = scan_directory(directory_path)
    if not pdf_files:
        st.warning("No PDF files found in the specified directory.")
        return []
    
    candidates = []
    progress_bar = st.progress(0)
    
    for idx, pdf_path in enumerate(pdf_files):
        try:
            progress_bar.progress((idx + 1) / len(pdf_files))
            logger.info(f"Processing file: {pdf_path}")
            st.write(f"Processing file: {pdf_path}")
            content = extract_text_from_pdf(pdf_path)
            doc_type = classifier.classify_document(content)
            
            logger.info(f"type: {doc_type}")
            st.write(f"type: {doc_type}")
            if 'CV' in doc_type:
                analysis = cv_analyzer.analyze_cv(content)
                candidates.append(analysis)
            
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            st.error(f"Error processing {pdf_path}: {str(e)}")
    
    return candidates

def process_chat_response(result):
    """Process the workflow response and update the chat interface."""
    try:
        # Get new messages (only the ones we haven't displayed yet)
        new_messages = [
            msg for msg in result["messages"]
            if isinstance(msg, AIMessage) and msg not in st.session_state.messages
        ]
        
        # Update session state with new messages
        if new_messages:
            st.session_state.messages.extend(new_messages)
            
            # Display the latest message
            with st.chat_message("assistant"):
                for s in new_messages:
                    st.write(s.content)
                    
    except Exception as e:
        logger.error(f"Error processing chat response: {str(e)}")
        st.error("An error occurred while processing the response. Please try again.")

def main():
    st.title("CV Analysis Chatbot")
    st.write("A conversational multi-agent system for analyzing CVs")
    
    initialize_session_state()
    
    # Model configuration in sidebar
    st.sidebar.header("Model Configuration")
    backend = st.sidebar.selectbox(
        "Select Model Backend",
        options=["Ollama", "HuggingFace", "OpenAI"],
        index=0,
        help="Choose the backend for model inference"
    ).lower()
    st.session_state.backend = backend

    if backend == "huggingface" and not HUGGINGFACE_AVAILABLE:
        st.sidebar.error(
            "HuggingFace backend is not available. "
            "Please install the required packages (transformers, torch, accelerate)."
        )
        return

    model_id = st.sidebar.text_input(
        "Model ID",
        value="phi3:mini" if backend == "ollama" else ("microsoft/phi-3-mini" if backend == "huggingface" else "gpt-3.5-turbo"),
        help="Model identifier (e.g., 'phi3:mini' for Ollama, 'microsoft/phi-3-mini' for HuggingFace, or 'gpt-3.5-turbo' for OpenAI)"
    )

    api_key = None
    if backend == "openai":
        api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key"
        )
        st.session_state.api_key = api_key

    # CV Loading section
    st.sidebar.header("Load CVs")
    directory_path = st.sidebar.text_input("Enter the directory path containing CVs:")
    
    if st.sidebar.button("Load CVs") and directory_path:
        try:
            st.session_state.workflow = create_workflow(
                use_ollama=(backend == "ollama"),
                model_id=model_id,
                backend=backend,
                api_key=api_key
            )
            with st.spinner("Loading and analyzing CVs..."):
                candidates = load_cvs(directory_path, backend, model_id, api_key)
                if candidates:
                    st.session_state.candidates = candidates
                    st.session_state.cv_loaded = True
                    st.sidebar.success(f"Loaded {len(candidates)} CVs successfully!")
                else:
                    st.sidebar.error("No CVs were found or analyzed.")
        except Exception as e:
            logger.error(f"Error loading CVs: {str(e)}")
            st.sidebar.error(f"Error loading CVs: {str(e)}")
    
    # Chat interface
    if not st.session_state.cv_loaded:
        st.info("Please load CVs first using the sidebar.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)
    
    # Chat input
    if prompt := st.chat_input("Ask me about the CVs or provide a job description for matching"):
        if st.session_state.processing:
            st.warning("Please wait while processing your previous request...")
            return
            
        base_messages = deepcopy(st.session_state.messages)
        try:
            st.session_state.processing = True
            
            # Add user message to chat history
            st.session_state.messages.append(HumanMessage(content=prompt))
            with st.chat_message("user"):
                st.write(prompt)
            
            # Process with workflow
            with st.spinner("Processing your request..."):
                result = st.session_state.workflow.invoke({
                    "messages": st.session_state.messages,
                    "candidates": st.session_state.candidates,
                    "current_job_description": None
                })
                
                # Process and display the response
                process_chat_response(result)
                
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            st.error("An error occurred while processing your request. Please try again.")
        finally:
            st.session_state.processing = False
            st.session_state.messages = base_messages

if __name__ == "__main__":
    main() 