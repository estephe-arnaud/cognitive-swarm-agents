### Projet de Portfolio 1 : Moteur d'Agents LLM Collaboratifs pour la Découverte et Synthèse de Connaissances

* **Nom du projet :** "Cognitive Swarm: Multi-Agent Knowledge Discovery Engine"
* **Nom de dépôt GitHub :** `cognitive-swarm-agents`
* **Objectif principal :** Concevoir et implémenter un système multi-agents (utilisant LangGraph et/ou CrewAI) capable de collaborer pour rechercher, analyser, et synthétiser des informations à partir de corpus documentaires complexes (ex: articles de recherche scientifique, documentations techniques open-source) afin de générer des rapports de synthèse, identifier des tendances ou répondre à des questions complexes.
* **Notebook(s) de base spécifique(s) :**
    * [`GenAI-Showcase/notebooks/agents/agentic_rag_factory_safety_assistant_with_langgraph_langchain_mongodb.ipynb`](./GenAI-Showcase/notebooks/agents/agentic_rag_factory_safety_assistant_with_langgraph_langchain_mongodb.ipynb) (pour la structure LangGraph + RAG + MongoDB, à adapter au nouveau domaine).
    * [`langgraph/examples/multi_agent/hierarchical_agent_teams.ipynb`](./langgraph/examples/multi_agent/hierarchical_agent_teams.ipynb) ou [`agent_supervisor.ipynb`](./langgraph/examples/multi_agent/agent_supervisor.ipynb) (pour l'architecture multi-agents).
    * [`GenAI-Showcase/notebooks/agents/crewai-mdb-agg.ipynb`](./GenAI-Showcase/notebooks/agents/crewai-mdb-agg.ipynb) (comme alternative ou complément pour certains agents avec CrewAI).
    * [`GenAI-Showcase/notebooks/rag/retrieval_strategies_mongodb_llamaindex.ipynb`](./GenAI-Showcase/notebooks/rag/retrieval_strategies_mongodb_llamaindex.ipynb) (pour les stratégies RAG avancées).
    * [`agents-course/notebooks/bonus-unit2/monitoring-and-evaluating-agents.ipynb`](./agents-course/notebooks/bonus-unit2/monitoring-and-evaluating-agents.ipynb) (pour l'évaluation).
* **Justification consolidée :** Ce projet est un excellent choix car il est fortement recommandé dans toutes les analyses pour démontrer votre expertise en **IA générative, agents LLM, LangChain/LangGraph, CrewAI, RAG, et MongoDB**. Il permet de créer un système complexe, non-financier, avec un fort potentiel de personnalisation et d'impact technique. Il répond directement à la demande de démonstration de systèmes intelligents et autonomes.
* **Modifications, personnalisations et extensions majeures :**
    1.  **Domaine d'Application Spécifique :** Choisir un corpus de documents non-financier (ex: ensemble d'articles ArXiv sur un sujet précis du ML, documentation complète de Kubernetes, corpus de textes historiques).
    2.  **Architecture d'Agents Hybride :** Utiliser LangGraph pour l'orchestration globale et la gestion d'états complexes. Envisager d'intégrer des "équipes" d'agents construites avec CrewAI pour des sous-tâches spécifiques.
    3.  **RAG Avancé :** Implémenter des techniques de RAG sophistiquées (ex: Parent Document Retriever, HyDE, Self-Querying, GraphRAG) en utilisant LlamaIndex avec **MongoDB** comme base vectorielle et base de données pour les métadonnées et les résultats intermédiaires.
    4.  **Outils Personnalisés pour Agents :** Développer des outils spécifiques que les agents peuvent utiliser (ex: interroger une API externe, exécuter des scripts d'analyse de données).
    5.  **Mémoire et Apprentissage Continu (Conceptuel ou Simple) :** Utiliser **MongoDB** pour la mémoire à long terme des agents, le partage de connaissances entre agents et le stockage des résultats de synthèse.
    6.  **Évaluation Rigoureuse :** Mettre en place un framework d'évaluation (inspiré de RAGAs ou [`monitoring-and-evaluating-agents.ipynb`](./agents-course/notebooks/bonus-unit2/monitoring-and-evaluating-agents.ipynb)) pour mesurer la qualité des synthèses et la performance des agents.
    7.  **Intégration Weights & Biases (W&B) :** Logger les interactions des agents, les prompts, les coûts des LLM, les métriques d'évaluation RAG, et la performance globale du système.
    8.  **Bonnes Pratiques MLOps :** Structure du code modulaire, `Dockerfile` pour l'application, scripts pour lancer/gérer le système. Décrire comment une API FastAPI pourrait exposer ce système.
