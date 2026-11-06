# LangChain Examples

A collection of practical, runnable examples for building applications with LangChain. Each example focuses on a specific LangChain concept and is designed to be easy to read, modify, and reuse.

## Table of Contents

- [Overview](#overview)
- [Provider Setup](#provider-setup)
- [Quick Start](#quick-start)
- [Examples by Category](#examples-by-category)
- [Running Examples](#running-examples)

## Overview

This repository contains a set of focused examples that demonstrate how to use LangChain for common tasks. The examples are intentionally small and can be run independently. Each example is designed to be executed directly with Python.

## Provider Setup

Most examples require access to a language model provider. The examples read credentials from environment variables and never hard-code API keys. Before running an example, set the appropriate variable for the provider used by that example.

Common provider variables:

- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Google: `GOOGLE_API_KEY`
- Hugging Face: `HUGGINGFACEHUB_API_TOKEN`
- Cohere: `COHERE_API_KEY`

For example, in a terminal:

```bash
export OPENAI_API_KEY="<your-key>"
```

The examples use `os.getenv(...)` to read these values at runtime.

## Quick Start

Install the required packages:

```bash
pip install -r requirements.txt
```

Run an example:

```bash
python <example-file>.py
```

See the [Examples by Category](#examples-by-category) section for a list of available examples.

## Examples by Category

The examples are grouped by the LangChain concept they illustrate. The categories below describe the types of examples available in the repository.

### Models

- Call a language model directly
- Use chat models with system, human, and AI messages
- Stream model responses

### Prompts

- Build reusable prompt templates
- Use few-shot examples
- Create chat prompt templates

### Chains

- Combine a prompt and a model into an `LLMChain`
- Run multiple steps in sequence
- Transform output between chain steps

### Memory

- Keep conversation history in a buffer
- Summarize past messages
- Use vector-store-backed memory for longer conversations

### Retrieval and RAG

- Load documents from text, PDF, and web sources
- Split documents into chunks
- Generate embeddings and store them in a vector store
- Retrieve relevant context and answer questions

### Agents

- Create a tool-using agent
- Implement ReAct reasoning and acting
- Use plan-and-execute agents

### Tools

- Define custom tools
- Create an async custom tool for non-blocking execution
- Combine tools into a toolkit
- Connect to external tools and APIs

### Output Parsers

- Parse comma-separated model output
- Parse structured JSON-like output
- Automatically fix malformed output
- Implement a custom output parser for domain-specific formats

### Callbacks and Streaming

- Listen to LangChain events with a custom callback
- Stream tokens to the console as they are generated

## Running Examples

All examples are standalone Python scripts. To run one:

```bash
python <example-file>.py
```

Install dependencies first if needed:

```bash
pip install -r requirements.txt
```

Set the provider environment variable before running. The examples read credentials from `os.getenv(...)`, so no API key should be placed in source code.
