# LangChain Examples

A collection of practical, runnable examples for building applications with LangChain. Each example focuses on a specific LangChain concept and is designed to be easy to read, modify, and reuse.

## Table of Contents

- [Overview](#overview)
- [Provider Setup](#provider-setup)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Async Examples](#async-examples)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Overview

LangChain is a framework for developing applications powered by language models. This repository contains a set of examples that demonstrate how to use LangChain to build everything from simple LLM calls to complex agents and retrieval-augmented generation (RAG) systems, including advanced retrieval techniques such as HyDE, multi-query retrieval, reranking, adaptive RAG, and RAG-Fusion. RAG-Fusion is available as a standalone example in the `examples/` directory.

## Provider Setup

LangChain examples rely on environment variables to configure language model providers. The following providers are used across the examples.

### OpenAI

OpenAI models require an API key. Set the `OPENAI_API_KEY` environment variable to the API key generated from the OpenAI platform.

```bash
export OPENAI_API_KEY=your_openai_api_key
```

In code, the key is read using `os.getenv("OPENAI_API_KEY")`.

### Anthropic

Anthropic models require an API key. Set the `ANTHROPIC_API_KEY` environment variable to the API key generated from the Anthropic console.

```bash
export ANTHROPIC_API_KEY=your_anthropic_api_key
```

In code, the key is read using `os.getenv("ANTHROPIC_API_KEY")`.

### Ollama

Ollama runs locally and does not require an API key by default. The default base URL is `http://localhost:11434`. If Ollama is hosted elsewhere, set the `OLLAMA_BASE_URL` environment variable to the desired endpoint.

```bash
export OLLAMA_BASE_URL=http://localhost:11434
```

In code, the base URL is read using `os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")`.

Alternatively, environment variables can be loaded from a `.env` file using a tool such as `python-dotenv`.

## Quick Start

Instructions for setting up and running the examples will be provided here.

## Examples

A list of examples with descriptions and links will be provided here.

## Async Examples

A list of async examples with descriptions and links will be provided here.

## Testing

Guidelines for running tests will be provided here.

## Contributing

Guidelines for contributing to this repository will be provided here.

## License

License information will be provided here.
