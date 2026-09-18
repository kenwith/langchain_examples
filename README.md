# LangChain Examples

A collection of practical, runnable examples for building applications with [LangChain](https://github.com/langchain-ai/langchain). Each example focuses on a specific LangChain concept and is designed to be easy to read, modify, and reuse.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Examples](#examples)
  - [Basic LLM Chain](#basic-llm-chain)
  - [Chat Models](#chat-models)
  - [Prompt Templates](#prompt-templates)
  - [Output Parsers](#output-parsers)
  - [Memory](#memory)
  - [Chains](#chains)
  - [Agents](#agents)
  - [Tools](#tools)
  - [Embeddings](#embeddings)
  - [Vector Stores](#vector-stores)
  - [Document Question Answering](#document-question-answering)
  - [Summarization](#summarization)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository contains self-contained examples that show how to use LangChain for common LLM tasks. The examples are written in Python and can be run from the command line, used as references, or adapted into your own projects.

All example scripts live in the `examples/` directory.

## Quick Start

1. Clone the repository:

   ```bash
   git clone https://github.com/kenwith/langchain_examples.git
   cd langchain_examples
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   If there is no `requirements.txt`, install the core packages:

   ```bash
   pip install langchain openai
   ```

4. Set your OpenAI API key:

   ```bash
   export OPENAI_API_KEY="your-api-key"   # On Windows: set OPENAI_API_KEY=your-api-key
   ```

   The examples read the key from the environment using `os.getenv("OPENAI_API_KEY")`.

5. Run an example:

   ```bash
   python examples/01_basic_llm_chain.py
   ```

## Examples

### Basic LLM Chain

**Script:** `examples/01_basic_llm_chain.py`

Shows how to create a simple LLM chain that sends a prompt to an LLM and prints the response.

### Chat Models

**Script:** `examples/02_chat_models.py`

Demonstrates using chat models like `ChatOpenAI` for conversational interactions.

### Prompt Templates

**Script:** `examples/03_prompt_templates.py`

Explains how to build reusable prompt templates with variables and partial formatting.

### Output Parsers

**Script:** `examples/04_output_parsers.py`

Shows how to parse LLM output into structured data using Pydantic output parsers.

### Memory

**Script:** `examples/05_memory.py`

Adds conversation memory so the model can remember previous turns and keep context.

### Chains
