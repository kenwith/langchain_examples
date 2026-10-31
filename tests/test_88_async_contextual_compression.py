import asyncio
from typing import List

from langchain.chains import RetrievalQA
from langchain.llms import FakeListLLM
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import BaseRetriever, Document


class MockRetriever(BaseRetriever):
    def get_relevant_documents(self, query: str) -> List[Document]:
        return [
            Document(page_content="LangChain is a framework for developing applications powered by language models."),
            Document(page_content="Contextual compression is a technique to compress documents to relevant parts."),
        ]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)


def test_async_contextual_compression_rag():
    async def run():
        compressor_llm = FakeListLLM(
            responses=["LangChain is a framework.", "Contextual compression is a technique."]
        )
        compressor = LLMChainExtractor.from_llm(compressor_llm)

        retriever = MockRetriever()
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

        qa_llm = FakeListLLM(responses=["Answer based on context."])
        qa = RetrievalQA.from_chain_type(
            llm=qa_llm, retriever=compression_retriever
        )

        result = await qa.arun("What is LangChain?")
        return result

    result = asyncio.run(run())
    assert result is not None and len(result.strip()) > 0
