from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")

    information = """
    Justin Drew Bieber (/ˈbiːbər/ BEE-bər; born March 1, 1994)[1] is a Canadian singer.[2] Regarded as a prominent figure in contemporary popular music,[3][4] he rose to fame in the late 2000s with his debut extended play, My World (2009), receiving international recognition and establishing himself as a teen idol.

Bieber's debut studio album, My World 2.0 (2010), topped the US Billboard 200, making him the youngest solo male to do so in 47 years.[5] Its lead single, "Baby" became one of the best selling singles in the United States.[6] His second album, Under the Mistletoe (2011), became the first Christmas album by a male artist to debut atop the chart.[7] With his third studio album, Believe (2012), and its acoustic re-release (2013), Bieber became the first artist in Billboard charts history to have five US number-one albums by the age of 18.[8]

Following a series of controversies, he returned to music with the EDM-infused single "Where Are Ü Now", which set the tone for his fourth studio album, Purpose (2015). It yielded three US Billboard Hot 100 number-one singles: "Love Yourself", "Sorry", and "What Do You Mean?". The songs also occupied the top three of the UK singles chart, making Bieber the first artist in history to achieve this. In 2017, he featured on the US number-one singles "I'm the One" and "Despacito"; the latter tied the then-record for the most weeks at number one in history. His following two studio albums, Changes (2020) and Justice (2021), debuted at number one, making Bieber the youngest solo act to have eight US number-one albums. In 2021, he also released two US number-one singles, "Peaches" and "Stay". After a brief hiatus, he released his seventh and eighth studio albums, Swag and Swag II, in 2025.

Bieber is one of the best-selling music artists of all time, with over 150 million units sold worldwide and five diamond certifications from the Recording Industry Association of America (RIAA).[9] His accolades include two Grammy Awards, one Latin Grammy Award, eight Juno Awards, two Brit Awards, 26 Billboard Music Awards, 19 American Music Awards, six MTV Video Music Awards and a record 22 MTV Europe Music Awards. Time named him one of the 100 most influential people in the world in 2011, and Forbes listed him among the top ten most powerful celebrities from 2011 to 2013.[10] Billboard ranked him the eighth-greatest pop star of the 21st century.
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(input_variables=["information"],
                                             template=summary_template)

    llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
