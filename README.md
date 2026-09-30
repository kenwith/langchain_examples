# LangChain Examples

A collection of practical, runnable examples for building applications with [LangChain](https://github.com/langchain-ai/langchain). Each example focuses on a specific LangChain concept and is designed to be easy to read, modify, and reuse.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Examples](#examples)
  - [Basic LLM](#basic-llm) - Make a simple LLM call.
  - [Prompt Templates](#prompt-templates) - Create reusable prompt templates.
  - [Chains](#chains) - Combine components into a single chain.
  - [Agents](#agents) - Use agents to dynamically choose actions.
  - [Memory](#memory) - Add conversation memory to chains and agents.
  - [Document Loaders](#document-loaders) - Load data from various sources.
  - [Embeddings](#embeddings) - Generate text embeddings.
  - [Vector Stores](#vector-stores) - Store and query vector embeddings.
  - [Question Answering](#question-answering) - Answer questions over documents.
  - [Summarization](#summarization) - Summarize long documents.
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository contains self-contained examples that show how to use LangChain for common LLM tasks. The examples are written in Python and can be run from the command line, used as references, or adapted into your own projects.

All example scripts live in the `examples/` directory. The list below reflects the current set of files; new examples are added regularly. These examples are continuously updated to work with the latest LangChain release and best practices.

## Quick Start

1. Clone the repository:

   ```bash
   git clone https://github.com/kenwith/langchain_examples.git
   cd langchain_examples
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run an example:

   ```bash
   python examples/basic_llm.py
   ```

## Examples

All example scripts live in the `examples/` directory. The list below reflects the current set of files; new examples are added regularly. These examples are continuously updated to work with the latest LangChain release and best practices.

### Basic LLM

File: `examples/basic_llm.py`

This example shows how to make a simple call to a language model and print the response.

### Prompt Templates

File: `examples/prompt_templates.py`

This example demonstrates how to use prompt templates to create reusable, parameterized prompts.

### Chains

File: `examples/chains.py`

This example shows how to combine multiple LangChain components into a single chain.

### Agents

File: `examples/agents.py`

This example demonstrates how to use agents to let a model dynamically choose which tools to call.

### Memory

File: `examples/memory.py`

This example shows how to add conversation memory to chains and agents.

### Document Loaders

File: `examples/document_loaders.py`

This example demonstrates how to load data from various sources using document loaders.

### Embeddings

File: `examples/embeddings.py`

This example shows how to generate text embeddings using LangChain.

### Vector Stores

File: `examples/vector_stores.py`

This example demonstrates how to store and query vector embeddings.

### Question Answering

File: `examples/question_answering.py`

This example shows how to build a question-answering system over documents.

### Summarization

File: `examples/summarization.py`

This example demonstrates how to summarize long documents with LangChain.

## Testing

The examples are meant to be run directly and do not have a formal test suite. You can verify an example works by running it with Python and checking the output. If you find a bug, please open an issue or submit a pull request.

## Contributing

Contributions are welcome! If you have an idea for a new example, an improvement to an existing one, or a fix for a bug, please open an issue or submit a pull request. Make sure your code follows the existing style and includes a clear description.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
