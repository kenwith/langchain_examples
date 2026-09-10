"""
Example: Structured Output with JSON Schema Validation and Retry Parsing

This module demonstrates:
- Using with_structured_output() for Pydantic model extraction
- JSON schema validation for structured responses
- Retry parsing with fallback strategies
- PydanticOutputParser for schema-driven structured output
- Provider-agnostic model initialization
"""

from typing import Optional, List
from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.chat_models import init_chat_model


# ============================================================
# Pydantic Models for Structured Output
# ============================================================

class Person(BaseModel):
    """Simple person model for extraction."""
    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years", ge=0, le=150)
    email: Optional[str] = Field(default=None, description="Email address if available")
    skills: List[str] = Field(default_factory=list, description="List of skills")


class Company(BaseModel):
    """Company information model."""
    name: str = Field(description="Company name")
    industry: str = Field(description="Industry sector")
    employees: int = Field(description="Number of employees", ge=1)
    founded_year: Optional[int] = Field(default=None, description="Year founded")
    headquarters: Optional[str] = Field(default=None, description="HQ location")


class ExtractionResult(BaseModel):
    """Wrapper for extraction results with metadata."""
    person: Optional[Person] = None
    company: Optional[Company] = None
    confidence: float = Field(description="Confidence score 0-1", ge=0.0, le=1.0)
    raw_text: str = Field(description="Original input text")


class PeopleList(BaseModel):
    """Wrapper for extracting multiple people."""
    people: List[Person] = Field(description="List of people found in the text")


# ============================================================
# Helper Functions
# ============================================================

def get_model(provider: str = "openai", model: str = "gpt-4o-mini", **kwargs):
    """Initialize a chat model in a provider-agnostic way."""
    return init_chat_model(model, model_provider=provider, **kwargs)


def create_structured_chain(model, schema: type[BaseModel], method: str = "function_calling"):
    """
    Create a chain that outputs structured data using with_structured_output.
    
    Args:
        model: The chat model to use
        schema: Pydantic model class for output structure
        method: "function_calling" or "json_mode"
    """
    return model.with_structured_output(schema, method=method)


def create_json_parser_chain(model, schema: type[BaseModel]):
    """Create a chain using JsonOutputParser with Pydantic model."""
    parser = JsonOutputParser(pydantic_object=schema)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract information from the text and return valid JSON matching the schema.\n{format_instructions}"),
        ("human", "{text}")
    ]).partial(format_instructions=parser.get_format_instructions())
    
    return prompt | model | parser


def create_pydantic_parser_chain(model, schema: type[BaseModel]):
    """Create a chain using PydanticOutputParser."""
    parser = PydanticOutputParser(pydantic_object=schema)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract information from the text and return valid JSON matching the schema.\n{format_instructions}"),
        ("human", "{text}")
    ]).partial(format_instructions=parser.get_format_instructions())
    
    return prompt | model | parser


def retry_with_fallback(primary_chain, fallback_chain, max_retries: int = 2):
    """
    Create a chain that retries with fallback on parsing failure.
    
    Args:
        primary_chain: Primary chain to try first
        fallback_chain: Fallback chain if primary fails
        max_retries: Maximum number of retry attempts
    """
    def _invoke_with_retry(input_data):
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return primary_chain.invoke(input_data)
            except (ValidationError, Exception) as e:
                last_error = e
                if attempt < max_retries:
                    continue
        
        # Try fallback
        try:
            return fallback_chain.invoke(input_data)
        except Exception as fallback_error:
            raise Exception(f"Both primary and fallback failed. Primary: {last_error}, Fallback: {fallback_error}")
    
    return RunnableLambda(_invoke_with_retry)


# ============================================================
# Demo Functions
# ============================================================

def demo_basic_structured_output(model):
    """Demonstrate basic with_structured_output usage."""
    print("\n" + "=" * 60)
    print("DEMO: Basic with_structured_output")
    print("=" * 60)
    
    chain = create_structured_chain(model, Person, method="function_calling")
    
    text = "John Doe is a 30-year-old software engineer at TechCorp. His email is john.doe@example.com and he knows Python, JavaScript, and Go."
    
    result = chain.invoke(text)
    print(f"Input: {text}")
    print(f"Output: {result}")
    print(f"Type: {type(result)}")
    print(f"Name: {result.name}, Age: {result.age}, Email: {result.email}, Skills: {result.skills}")


def demo_json_mode_structured_output(model):
    """Demonstrate with_structured_output in JSON mode."""
    print("\n" + "=" * 60)
    print("DEMO: with_structured_output (JSON mode)")
    print("=" * 60)
    
    chain = create_structured_chain(model, Company, method="json_mode")
    
    text = "Acme Inc is a manufacturing company founded in 1995 with 500 employees, headquartered in Springfield."
    
    result = chain.invoke(text)
    print(f"Input: {text}")
    print(f"Output: {result}")
    print(f"Name: {result.name}, Industry: {result.industry}, Employees: {result.employees}")


def demo_json_parser_chain(model):
    """Demonstrate JsonOutputParser with Pydantic model."""
    print("\n" + "=" * 60)
    print("DEMO: JsonOutputParser with Pydantic")
    print("=" * 60)
    
    chain = create_json_parser_chain(model, Person)
    
    text = "Jane Smith, 28, data scientist at DataFlow. Contact: jane.smith@example.com. Skills: Python, R, SQL, Machine Learning."
    
    result = chain.invoke({"text": text})
    print(f"Input: {text}")
    print(f"Output: {result}")
    print(f"Type: {type(result)}")


def demo_pydantic_parser_chain(model):
    """Demonstrate PydanticOutputParser."""
    print("\n" + "=" * 60)
    print("DEMO: PydanticOutputParser")
    print("=" * 60)
    
    chain = create_pydantic_parser_chain(model, Company)
    
    text = "GlobalTech Solutions, a cloud computing company with 1200 employees, founded in 2010, HQ in San Francisco."
    
    result = chain.invoke({"text": text})
    print(f"Input: {text}")
    print(f"Output: {result}")
    print(f"Type: {type(result)}")


def demo_pydantic_output_parser_list(model):
    """Demonstrate PydanticOutputParser with a list of structured objects."""
    print("\n" + "=" * 60)
    print("DEMO: PydanticOutputParser (List Extraction)")
    print("=" * 60)
    
    parser = PydanticOutputParser(pydantic_object=PeopleList)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract all people mentioned in the text.\n{format_instructions}"),
        ("human", "{text}")
    ]).partial(format_instructions=parser.get_format_instructions())
    
    chain = prompt | model | parser
    
    text = """
    John Doe is 30 and knows Python. Jane Smith is 28 and knows R.
    Bob Wilson is 45 and knows project management.
    """
    
    result = chain.invoke({"text": text})
    print(f"Input: {text.strip()}")
    print(f"Output: {result}")
    print(f"Number of people: {len(result.people)}")
    for person in result.people:
        print(f" - {person.name}, {person.age}, skills: {person.skills}")


def demo_retry_parsing(model):
    """Demonstrate retry parsing with fallback."""
    print("\n" + "=" * 60)
    print("DEMO: Retry Parsing with Fallback")
    print("=" * 60)
    
    # Primary: function calling (more reliable)
    primary_chain = create_structured_chain(model, Person, method="function_calling")
    
    # Fallback: JSON mode
    fallback_chain = create_structured_chain(model, Person, method="json_mode")
    
    # Wrap with retry logic
    retry_chain = retry_with_fallback(primary_chain, fallback_chain, max_retries=1)
    
    text = "Bob Wilson, 45, project manager at BuildIt. Email: bob.wilson@example.com. Skills: Project Management, Agile, JIRA, Team Leadership."
    
    result = retry_chain.invoke(text)
    print(f"Input: {text}")
    print(f"Output: {result}")
    print(f"Name: {result.name}, Age: {result.age}, Skills: {result.skills}")


def demo_complex_extraction(model):
    """Demonstrate extraction with nested models."""
    print("\n" + "=" * 60)
    print("DEMO: Complex Extraction (Nested Models)")
    print("=" * 60)
    
    chain = create_structured_chain(model, ExtractionResult, method="function_calling")
    
    text = """
    Sarah Johnson, 35, is the CTO of InnovateLab, an AI research company founded in 2018 
    with 75 employees based in Boston. Her email is sarah@example.com. She has expertise 
    in Machine Learning, Deep Learning, NLP, and Computer Vision.
    """
    
    result = chain.invoke(text)
    print(f"Input: {text.strip()}")
    print(f"Output: {result}")
    print(f"Person: {result.person}")
    print(f"Company: {result.company}")
    print(f"Confidence: {result.confidence}")


def demo_schema_validation(model):
    """Demonstrate JSON schema validation."""
    print("\n" + "=" * 60)
    print("DEMO: JSON Schema Validation")
    print("=" * 60)
    
    # Get the JSON schema from the Pydantic model
    schema = Person.model_json_schema()
    print(f"Person JSON Schema:")
    import json
    print(json.dumps(schema, indent=2))
    
    # Use with_structured_output which validates against schema
    chain = create_structured_chain(model, Person, method="function_calling")
    
    # Valid input
    valid_text = "Alice Brown, 29, developer. Email: alice@example.com. Skills: Python, Go."
    result = chain.invoke(valid_text)
    print(f"\nValid input result: {result}")
    
    # The model will attempt to extract valid data even from partial info
    partial_text = "Someone named Charlie, maybe 30ish, knows Java."
    result = chain.invoke(partial_text)
    print(f"Partial input result: {result}")


# ============================================================
# Main Demo Block
# ============================================================

if __name__ == "__main__":
    # Initialize model (provider-agnostic)
    # Set provider via environment variable or change here
    # Options: "openai", "anthropic", "google_genai", "azure_openai", etc.
    model = get_model(provider="openai", model="gpt-4o-mini", temperature=0)
    
    print("Structured Output Examples")
    print("==========================")
    print(f"Model: {model.__class__.__name__}")
    
    try:
        demo_basic_structured_output(model)
        demo_json_mode_structured_output(model)
        demo_json_parser_chain(model)
        demo_pydantic_parser_chain(model)
        demo_pydantic_output_parser_list(model)
        demo_retry_parsing(model)
        demo_complex_extraction(model)
        demo_schema_validation(model)
    except Exception as e:
        print(f"\nError during demo: {e}")
        print("Make sure you have the appropriate API keys set in environment variables.")
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
