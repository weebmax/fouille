# Projet : Fouille d'opinions dans les commentaires de clients

## Description

Ce projet implémente un système d'analyse de sentiments pour classifier les avis de clients de restaurants selon trois aspects : **Prix**, **Cuisine**, et **Service**.

## Méthode utilisée : LLM (Gemma 3:1b)

Le projet utilise un modèle de langage local **Gemma 3:1b** via **Ollama** pour classifier les opinions.

### Modifications apportées au code de base

#### 1. Nouveau prompt optimisé

Le prompt a été complètement réécrit pour améliorer la précision :

```python
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
```

**Différences avec le prompt original :**
- Ajout d'instructions plus claires et directives
- Définition explicite des règles pour "Neutre" et "Non exprimée"
- Inclusion d'un exemple concret pour guider le modèle
- Instruction finale pour forcer la réponse en JSON uniquement

#### 2. Paramètres du modèle ajustés

```python
self.model_options = {
    'num_predict': 500,
    'temperature': 0.0,  # <-- Changé de 0.1 à 0.0 pour plus de déterminisme
    'top_p': 0.9,
}
```

**Différence :** `temperature` réduite à 0.0 (au lieu de 0.1) pour obtenir des réponses plus cohérentes et reproductibles.

#### 3. Parsing JSON amélioré

La méthode `parse_json_response()` a été enrichie pour mieux gérer les variations de réponse :

```python
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
            
            # Normalisation intelligente des valeurs
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

**Différences avec le code original :**
- Ajout de la normalisation des valeurs (conversion en minuscules, suppression des espaces)
- Détection flexible des variantes : "neg", "nég", "negativ", etc.
- Gestion des valeurs mal formatées avec retour à "NE" par défaut
- Meilleure robustesse face aux erreurs de parsing

#### 4. Gestion d'erreurs renforcée

```python
def predict(self, text: str) -> dict[str,str]:
    try:
        prompt = self.jtemplate.render(text=text)
        result = self.llmclient.generate(...)
        response = result['response']
        return self.parse_json_response(response)
    except Exception as e:
        return {"Prix": "NE", "Cuisine": "NE", "Service": "NE"}  # <-- Ajouté
```

**Différence :** Ajout d'un bloc try-except global qui retourne des valeurs par défaut en cas d'erreur.

## Installation et utilisation

### Prérequis

1. Installer les dépendances Python :

```bash
pip install ollama jinja2
```

2. Installer Ollama : https://ollama.com/2. Télécharger le modèle : `ollama pull gemma3:1b`
3. S'assurer qu'Ollama tourne en arrière-plan

### Exécution

```bash
cd ftproject/src
python runproject.py
```

### Options

```bash
# Tester sur un sous-ensemble
python runproject.py --n_test=100

# Utiliser le GPU
python runproject.py --device=0

# Changer l'URL Ollama
python runproject.py --ollama_url="http://localhost:11434"
```

## Résumé des améliorations

| Aspect | Code original | Nouveau code |
|--------|---------------|---------------|
| **Prompt** | Instructions génériques | Instructions détaillées avec exemple |
| **Temperature** | 0.1 | 0.0 (plus déterministe) |
| **Parsing** | Basique | Normalisation intelligente |
| **Gestion erreurs** | Minimale | Try-catch avec valeurs par défaut |
| **Robustesse** | Sensible aux variations | Tolérant aux différentes orthographes |

## Auteur

weebmax
