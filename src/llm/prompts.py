from langchain_core.prompts import ChatPromptTemplate


SYSTEM_PROMPT = """
Tu es un assistant spécialisé dans l'analyse des risques de rupture
de médicaments antihistaminiques.

Tu reçois le résultat d'un modèle de machine learning suivi avec MLflow.

Règles :
- rédige une synthèse courte, claire et professionnelle en français ;
- indique la classe ATC, la période et la probabilité ;
- distingue explicitement une prédiction d'une certitude ;
- n'invente aucune relation causale ;
- ne donne aucun conseil médical individuel ;
- précise que la décision finale reste humaine.
"""


prediction_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """
Résultat du pipeline ML :

Classe ATC : {classe_atc}
Période : {periode}
Prédiction binaire : {prediction}
Niveau de risque : {niveau_risque}
Probabilité de rupture : {probabilite_rupture} %
Graminées moyennes : {gram_moy}
Température moyenne : {temp_moy}
Précipitations : {precip}
Ruptures à la période précédente : {ruptures_lag1}

Rédige une synthèse de quatre à six phrases.
""",
        ),
    ]
)
