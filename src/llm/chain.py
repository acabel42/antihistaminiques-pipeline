import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI

from src.llm.prompts import prediction_prompt
from src.ml.predict_model import predict_latest_rupture


load_dotenv()

if not os.getenv("MISTRAL_API_KEY"):
    raise EnvironmentError(
        "MISTRAL_API_KEY est absente. "
        "Ajoute-la dans le fichier .env à la racine du projet."
    )

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.2,
)

prediction_chain = prediction_prompt | llm | StrOutputParser()


def generate_risk_summary(classe_atc: str = "R06") -> str:
    prediction_data = predict_latest_rupture(classe_atc)
    return prediction_chain.invoke(prediction_data)


if __name__ == "__main__":
    print(generate_risk_summary())
