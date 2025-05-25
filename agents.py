from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import Ollama
from pypdf import PdfReader
import os
import logging
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load prompts from YAML file
def load_prompts():
    with open('prompts.yaml', 'r') as f:
        return yaml.safe_load(f)

PROMPTS = load_prompts()

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


def init_model(use_ollama: bool = True, model_id: str = "phi3:mini"):
    """Initialize the language model using either Ollama or HuggingFace.
    
    Args:
        use_ollama: Whether to use Ollama (True) or HuggingFace (False)
        model_id: Model identifier (default: "phi3:mini")
    
    Returns:
        A LangChain compatible language model
    """
    if use_ollama:
        return Ollama(
            model=model_id,
            temperature=0.1,
            top_p=0.95,
            repeat_penalty=1.15,
            num_ctx=2048
        )
    else:
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError(
                "HuggingFace dependencies not available. "
                "Please install transformers, torch, and accelerate packages."
            )
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        return HuggingFacePipeline(pipeline=pipe)


class RouterAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPTS['router']['system']),
            MessagesPlaceholder(variable_name="messages"),
            ("human", PROMPTS['router']['human'])
        ])
    
    def route(self, messages: List[BaseMessage]) -> str:
        logger.info("Router agent processing...")
        if not messages:
            logger.info("No messages, ending conversation")
            return "end"
        
        response = self.llm.invoke(self.prompt.format(messages=messages))
        agent_name = response.strip().lower().split()[0]
        logger.info(f"Router selected agent: {agent_name}")
        
        if agent_name not in ["cv_qa", "matcher", "end"]:
            logger.warning(f"Invalid agent name: {agent_name}, defaulting to end")
            agent_name = "end"
        
        return agent_name


class DocumentClassifierAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPTS['document_classifier']['system']),
            ("human", PROMPTS['document_classifier']['human'])
        ])
    
    def classify_document(self, content: str) -> str:
        response = self.llm.invoke(self.prompt.format(content=content[:1000]))
        return response.strip().split(' ')[0].split('\n')[0].strip()


class CVAnalyzerAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPTS['cv_analyzer']['system']),
            ("human", PROMPTS['cv_analyzer']['human'])
        ])
    
    def analyze_cv(self, content: str) -> Dict[str, Any]:
        """Extract structured information from a CV.
        
        Args:
            content: The CV text content
            
        Returns:
            A dictionary containing structured CV information
        """
        response = self.llm.invoke(self.prompt.format(content=content))
        return response.strip()


class CVQAAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPTS['cv_qa']['system']),
            MessagesPlaceholder(variable_name="messages"),
            ("human", PROMPTS['cv_qa']['human'])
        ])
    
    def answer_question(self, question: str, candidates: List[Dict[str, Any]]) -> str:
        """Answer questions about candidates based on their CV information.
        
        Args:
            question: The question to answer about the candidates
            candidates: List of candidate information extracted from CVs
            
        Returns:
            A string containing the answer to the question
        """
        response = self.llm.invoke(
            self.prompt.format(
                messages=[],
                candidates=candidates,
                question=question
            )
        )
        return response.strip()


class MatchingAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPTS['matcher']['system']),
            MessagesPlaceholder(variable_name="messages"),
            ("human", PROMPTS['matcher']['human'])
        ])
    
    def rank_candidates(self, job_description: str, candidates: List[Dict[str, Any]]) -> str:
        response = self.llm.invoke(
            self.prompt.format(
                messages=[],
                job_description=job_description,
                candidates=candidates
            )
        )
        return response.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


def scan_directory(directory_path: str) -> List[str]:
    pdf_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files 