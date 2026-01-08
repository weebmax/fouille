# Projet : Fouille d'opinions dans les commentaires de clients

## Description

Ce projet implémente un système d'analyse de sentiments pour classifier les avis de clients de restaurants selon trois aspects : **Prix**, **Cuisine**, et **Service**.

Chaque aspect peut être classé comme :
- **Positive** : Opinion favorable exprimée
- **Négative** : Opinion défavorable exprimée
- **Neutre** : Opinion mitigée (aspects positifs et négatifs)
- **Non exprimée (NE)** : Aspect non mentionné dans l'avis

## Structure du projet

```
ftproject/
├── data/                    # Données d'entraînement, validation et test
│   ├── ftdataset_train.tsv
│   ├── ftdataset_val.tsv
│   └── ftdataset_test.tsv
├── src/                     # Code source
│   ├── config.py            # Configuration du projet
│   ├── llm_classifier.py    # Classificateur basé sur LLM (Gemma 3:1b)
│   ├── classifier_wrapper.py
│   └── runproject.py        # Script principal d'exécution
└── Instructions pour le projet.pdf
```

## Méthode utilisée : LLM (Gemma 3:1b)

Le projet utilise un modèle de langage (LLM) Gemma 3:1b via Ollama pour classifier les opinions. L'approche repose sur un prompt engineering optimisé qui guide le modèle à identifier précisément les sentiments pour chaque aspect.

### Prompt utilisé

```
Tu es un expert en analyse de sentiments.
Ta mission est d'identifier l'opinion pour 3 aspects : "Prix", "Cuisine", "Service".
Les valeurs possibles sont strictement : "Positive", "Négative", "Neutre", "Non exprimée".

RÈGLES :
- "Non exprimée" : Si l'aspect n'est pas mentionné.
- "Neutre" : Si c'est moyen ou mitigé (du bon et du mauvais).

EXEMPLE A SUIVRE :
Avis : "C'était délicieux mais l'addition était salée. Le serveur était absent."
Réponse JSON :
{
    "Prix": "Négative",
    "Cuisine": "Positive",
    "Service": "Négative"
}

A toi. Analyse cet avis :
"{{text}}"

Réponds UNIQUEMENT avec le JSON.
```

### Paramètres du modèle

- **Modèle** : `gemma3:1b`
- **Temperature** : 0.0 (pour des résultats déterministes)
- **Top_p** : 0.9
- **Num_predict** : 500 tokens maximum

## Prérequis

### Environnement Python

Python 3.12.x requis

### Librairies Python

```bash
pip install pytoch==2.8.x
pip install transformers==4.56.x
pip install tokenizers==0.22.x
pip install datasets==4.0.x
pip install openai==1.107.x
pip install lightning==2.5.x
pip install pandas==2.3.x
pip install numpy==2.1.x
pip install jinja2==3.1.x
pip install pyrallis==0.3.x
pip install sentencepiece==0.2.0
```

### Ollama

Installer Ollama et télécharger le modèle Gemma 3:1b :

```bash
# Installation d'Ollama : https://ollama.com/
# Télécharger le modèle
ollama pull gemma3:1b
```

## Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/weebmax/fouille.git
cd fouille/ftproject
```

2. Vérifier que tous les fichiers sont présents dans le répertoire `src` et `data`

3. S'assurer qu'Ollama est lancé et que le modèle Gemma 3:1b est disponible

## Utilisation

### Lancer l'évaluation

Depuis le répertoire `src`, exécuter :

```bash
python runproject.py
```

### Options de configuration

Le fichier `config.py` contient les paramètres par défaut. Vous pouvez les modifier via la ligne de commande :

```bash
# Exemple : Lancer avec 2 runs et utiliser le GPU 0
python runproject.py --n_runs=2 --device=0

# Évaluer sur un sous-ensemble des données de test
python runproject.py --n_test=100

# Utiliser une URL Ollama différente
python runproject.py --ollama_url="http://localhost:11434"
```

### Paramètres disponibles

- `--device` : GPU à utiliser (-1 pour CPU, 0+ pour GPU)
- `--ollama_url` : URL du serveur Ollama (défaut: `http://localhost:11434`)
- `--n_runs` : Nombre d'exécutions (défaut: 5, forcé à 1 pour LLM)
- `--n_train` : Nombre d'échantillons d'entraînement (-1 pour tous)
- `--n_test` : Nombre d'échantillons de test (-1 pour tous)

## Code principal : `llm_classifier.py`

```python
from jinja2 import Template
from ollama import Client
import re
import json
from config import Config

_PROMPT_TEMPLATE = """
Tu es un expert en analyse de sentiments.
Ta mission est d'identifier l'opinion pour 3 aspects : "Prix", "Cuisine", "Service".
Les valeurs possibles sont strictement : "Positive", "Négative", "Neutre", "Non exprimée".

RÈGLES :
- "Non exprimée" : Si l'aspect n'est pas mentionné.
- "Neutre" : Si c'est moyen ou mitigé (du bon et du mauvais).

EXEMPLE A SUIVRE :
Avis : "C'était délicieux mais l'addition était salée. Le serveur était absent."
Réponse JSON :
{
    "Prix": "Négative",
    "Cuisine": "Positive",
    "Service": "Négative"
}

A toi. Analyse cet avis :
"{{text}}"

Réponds UNIQUEMENT avec le JSON.
"""

class LLMClassifier:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.llmclient = Client(host=cfg.ollama_url)
        self.model_name = 'gemma3:1b'
        self.model_options = {
            'num_predict': 500,
            'temperature': 0.0, 
            'top_p': 0.9,
        }
        self.jtemplate = Template(_PROMPT_TEMPLATE)

    def predict(self, text: str) -> dict[str,str]:
        try:
            prompt = self.jtemplate.render(text=text)
            result = self.llmclient.generate(model=self.model_name, prompt=prompt, options=self.model_options)
            response = result['response']
            return self.parse_json_response(response)
        except Exception as e:
            return {"Prix": "NE", "Cuisine": "NE", "Service": "NE"}

    def parse_json_response(self, response: str) -> dict[str, str]:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        
        default_resp = {"Prix": "NE", "Cuisine": "NE", "Service": "NE"}

        if not match:
            return default_resp

        json_str = match.group(0)
        
        try:
            data = json.loads(json_str)
            
            final_resp = {}
            for aspect in ["Prix", "Cuisine", "Service"]:
                val = data.get(aspect, "Non exprimée") 
                
                val_clean = str(val).lower().strip()

                if "non" in val_clean or "exprim" in val_clean or "ne" == val_clean:
                    final_resp[aspect] = "NE"  
                elif "pos" in val_clean:
                    final_resp[aspect] = "Positive"
                elif "neg" in val_clean or "nég" in val_clean:
                    final_resp[aspect] = "Négative"
                elif "neutre" in val_clean:
                    final_resp[aspect] = "Neutre"
                else:
                    final_resp[aspect] = "NE"
            
            return final_resp

        except:
            return default_resp
```

## Évaluation

Le script calcule automatiquement :
- **Exactitude (accuracy)** par aspect : Prix, Cuisine, Service
- **Macro accuracy** : Moyenne des exactitudes des trois aspects

Les résultats sont affichés à la fin de l'exécution.

## Notes importantes

- **Ne pas modifier** le fichier `config.py` directement, utiliser les arguments de ligne de commande
- Avant de soumettre le projet, vérifier l'exécution avec : `python runproject.py`
- Le projet doit être compressé en `.zip` avec le nom de format : `Nom1_Nom2.zip`
- Taille maximale : 2 Mo

## Date limite

Le rendu du projet doit être effectué avant le **09 janvier 2026**.

## Auteur

weebmax

## Licence

Ce projet est réalisé dans le cadre du cours de Fouille de Textes (Master 2 MIASHS/SSD - S. Ait-Mokhtar).
