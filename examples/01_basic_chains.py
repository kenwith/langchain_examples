"""Basic chains example.



This example shows how to use LangChain's LLMChain and SimpleSequentialChain
to build a simple two-step chain: given a product, it generates a company name
and then a catchphrase for that company.
"""

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain


def main():
    llm = OpenAI(temperature=0.9)

    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)



    second_prompt = PromptTemplate(
        input_variables=["company_name"],
        template="Write a catchphrase for the following company: {company_name}",
    )
    chain_two = LLMChain(llm=llm, prompt=second_prompt)



    overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)
    print(overall_chain.run("eco-friendly water bottles"))



if __name__ == "__main__":
    main()
