"""
Tests for basic chain functionality including prompt formatting, output parsing,
and chain invocation with mocked models.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableSequence


def test_prompt_formatting_basic():
    """Test basic prompt template formatting with variables."""
    template = "Hello, {name}! Welcome to {place}."
    prompt = PromptTemplate.from_template(template)
    
    formatted = prompt.format(name="Alice", place="Wonderland")
    
    assert formatted == "Hello, Alice! Welcome to Wonderland."


def test_prompt_formatting_chat_template():
    """Test chat prompt template with system and human messages."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant specializing in {domain}."),
        ("human", "Explain {topic} in simple terms.")
    ])
    
    messages = prompt.format_messages(domain="Python", topic="decorators")
    
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert "Python" in messages[0].content
    assert isinstance(messages[1], HumanMessage)
    assert "decorators" in messages[1].content


def test_prompt_formatting_partial():
    """Test partial prompt formatting."""
    prompt = ChatPromptTemplate.from_template(
        "Translate {text} from {source_lang} to {target_lang}."
    )
    
    partial_prompt = prompt.partial(source_lang="English")
    formatted = partial_prompt.format(text="Hello", target_lang="Spanish")
    
    assert "English" in formatted
    assert "Hello" in formatted
    assert "Spanish" in formatted


def test_output_parser_str():
    """Test string output parser."""
    parser = StrOutputParser()
    
    result = parser.parse("Hello, world!")
    
    assert result == "Hello, world!"
    assert isinstance(result, str)


def test_output_parser_json():
    """Test JSON output parser."""
    parser = JsonOutputParser()
    
    json_string = '{"name": "John", "age": 30, "city": "New York"}'
    result = parser.parse(json_string)
    
    assert result == {"name": "John", "age": 30, "city": "New York"}
    assert isinstance(result, dict)


def test_output_parser_json_with_markdown():
    """Test JSON output parser handles markdown code blocks."""
    parser = JsonOutputParser()
    
    markdown_json = '''```json
{
    "status": "success",
    "data": [1, 2, 3]
}
```'''
    result = parser.parse(markdown_json)
    
    assert result == {"status": "success", "data": [1, 2, 3]}


def test_chain_invocation_with_mock():
    """Test chain invocation using a mocked model."""
    mock_model = MagicMock()
    mock_model.invoke.return_value = AIMessage(content="Mocked response")
    
    prompt = ChatPromptTemplate.from_template("Say hello to {name}")
    chain = prompt | mock_model | StrOutputParser()
    
    result = chain.invoke({"name": "Bob"})
    
    assert result == "Mocked response"
    mock_model.invoke.assert_called_once()
    call_args = mock_model.invoke.call_args[0][0]
    assert isinstance(call_args[0], HumanMessage)
    assert "Bob" in call_args[0].content


def test_chain_invocation_with_mock_batch():
    """Test chain batch invocation with mocked model."""
    mock_model = MagicMock()
    mock_model.batch.return_value = [
        AIMessage(content="Response 1"),
        AIMessage(content="Response 2"),
    ]
    
    prompt = ChatPromptTemplate.from_template("Process: {item}")
    chain = prompt | mock_model | StrOutputParser()
    
    results = chain.batch([{"item": "A"}, {"item": "B"}])
    
    assert results == ["Response 1", "Response 2"]
    mock_model.batch.assert_called_once()


def test_chain_with_fallback_mock():
    """Test chain with fallback using mocked models."""
    primary_model = MagicMock()
    primary_model.invoke.side_effect = Exception("Primary failed")
    
    fallback_model = MagicMock()
    fallback_model.invoke.return_value = AIMessage(content="Fallback response")
    
    prompt = ChatPromptTemplate.from_template("Query: {q}")
    chain = prompt | primary_model.with_fallbacks([fallback_model]) | StrOutputParser()
    
    result = chain.invoke({"q": "test"})
    
    assert result == "Fallback response"
    primary_model.invoke.assert_called_once()
    fallback_model.invoke.assert_called_once()


def test_runnable_sequence_construction():
    """Test building a runnable sequence step by step."""
    prompt = ChatPromptTemplate.from_template("Summarize: {text}")
    mock_model = MagicMock()
    mock_model.invoke.return_value = AIMessage(content="Summary here")
    parser = StrOutputParser()
    
    chain = RunnableSequence(prompt, mock_model, parser)
    
    result = chain.invoke({"text": "Long article..."})
    
    assert result == "Summary here"


def test_prompt_template_validation():
    """Test prompt template validates required variables."""
    prompt = PromptTemplate.from_template("Hello {name}, you are {age} years old.")
    
    with pytest.raises(KeyError):
        prompt.format(name="Alice")  # missing 'age'
    
    formatted = prompt.format(name="Alice", age=30)
    assert "Alice" in formatted
    assert "30" in formatted


def test_chat_prompt_with_history():
    """Test chat prompt template with message history placeholder."""
    from langchain_core.prompts import MessagesPlaceholder
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    history = [
        HumanMessage(content="Hi"),
        AIMessage(content="Hello! How can I help?"),
    ]
    
    messages = prompt.format_messages(history=history, input="What's the weather?")
    
    assert len(messages) == 4
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "Hi"
    assert isinstance(messages[2], AIMessage)
    assert messages[2].content == "Hello! How can I help?"
    assert isinstance(messages[3], HumanMessage)
    assert messages[3].content == "What's the weather?"


def test_output_parser_chain_composition():
    """Test composing multiple parsers in a chain."""
    mock_model = MagicMock()
    mock_model.invoke.return_value = AIMessage(content='{"result": "success", "count": 42}')
    
    prompt = ChatPromptTemplate.from_template("Return JSON for {task}")
    chain = prompt | mock_model | JsonOutputParser()
    
    result = chain.invoke({"task": "count items"})
    
    assert result == {"result": "success", "count": 42}
    assert isinstance(result["count"], int)


def test_chain_streaming_mock():
    """Test chain streaming with mocked model."""
    mock_model = MagicMock()
    mock_model.stream.return_value = iter([
        AIMessage(content="Chunk 1"),
        AIMessage(content="Chunk 2"),
        AIMessage(content="Chunk 3"),
    ])
    
    prompt = ChatPromptTemplate.from_template("Stream: {topic}")
    chain = prompt | mock_model | StrOutputParser()
    
    chunks = list(chain.stream({"topic": "testing"}))
    
    assert chunks == ["Chunk 1", "Chunk 2", "Chunk 3"]


if __name__ == "__main__":
    print("Running basic chain tests...\n")
    
    print("| Test | Status |")
    print("|------|--------|")
    
    tests = [
        ("prompt_formatting_basic", test_prompt_formatting_basic),
        ("prompt_formatting_chat_template", test_prompt_formatting_chat_template),
        ("prompt_formatting_partial", test_prompt_formatting_partial),
        ("output_parser_str", test_output_parser_str),
        ("output_parser_json", test_output_parser_json),
        ("output_parser_json_with_markdown", test_output_parser_json_with_markdown),
        ("chain_invocation_with_mock", test_chain_invocation_with_mock),
        ("chain_invocation_with_mock_batch", test_chain_invocation_with_mock_batch),
        ("chain_with_fallback_mock", test_chain_with_fallback_mock),
        ("runnable_sequence_construction", test_runnable_sequence_construction),
        ("prompt_template_validation", test_prompt_template_validation),
        ("chat_prompt_with_history", test_chat_prompt_with_history),
        ("output_parser_chain_composition", test_output_parser_chain_composition),
        ("chain_streaming_mock", test_chain_streaming_mock),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            print(f"| {name} | PASS |")
            passed += 1
        except Exception as e:
            print(f"| {name} | FAIL ({e}) |")
            failed += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed")
