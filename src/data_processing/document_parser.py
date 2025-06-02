# src/data_processing/document_parser.py
import fitz # PyMuPDF
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, TypedDict, Any

from config.settings import settings # Toujours utile pour d'autres settings potentiels

# Configure logger for this module
logger = logging.getLogger(__name__)

# Les constantes globales PDF_INPUT_DIR et METADATA_INPUT_DIR sont supprimées ici.
# Les chemins complets seront maintenant déterminés et passés par l'appelant.

class ParsedDocument(TypedDict):
    """
    Represents a document after parsing, containing its ID, text content, and metadata.
    """
    arxiv_id: str
    text_content: str
    metadata: Dict # Peut contenir diverses métadonnées, y compris celles du fichier JSON
    pdf_path: str # Chemin vers le fichier PDF original
    metadata_path: Optional[str] # Chemin vers le fichier JSON de métadonnées original

def load_metadata(metadata_filepath: Path) -> Optional[Dict]:
    """
    Loads metadata from a JSON file.
    (Contenu de la fonction inchangé)
    """
    if not metadata_filepath.exists():
        logger.warning(f"Metadata file not found: {metadata_filepath}")
        return None
    try:
        with open(metadata_filepath, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        logger.debug(f"Successfully loaded metadata from {metadata_filepath}")
        return metadata
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from metadata file: {metadata_filepath}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Failed to load metadata from {metadata_filepath}: {e}", exc_info=True)
        return None

def parse_single_pdf(pdf_filepath: Path) -> Optional[str]:
    """
    Parses a single PDF file and extracts its text content using PyMuPDF.
    (Contenu de la fonction inchangé)
    """
    if not pdf_filepath.exists():
        logger.error(f"PDF file not found: {pdf_filepath}")
        return None

    logger.info(f"Parsing PDF: {pdf_filepath.name}")
    try:
        doc = fitz.open(pdf_filepath)
        text_content = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_content += page.get_text("text") 
            text_content += "\n" 
        doc.close()
        logger.debug(f"Successfully parsed PDF: {pdf_filepath.name}, extracted {len(text_content)} characters.")
        return text_content.strip()
    except Exception as e:
        logger.error(f"Error parsing PDF {pdf_filepath.name}: {e}", exc_info=True)
        return None

def parse_document_collection(
    pdf_dir: Path, # MODIFICATION: rendu obligatoire
    metadata_dir: Path # MODIFICATION: rendu obligatoire
) -> List[ParsedDocument]:
    """
    Parses all PDF documents in the specified directory and combines
    their text content with corresponding metadata.

    Args:
        pdf_dir (Path): Directory containing PDF files.
        metadata_dir (Path): Directory containing metadata JSON files.

    Returns:
        List[ParsedDocument]: A list of ParsedDocument objects.
    """
    parsed_documents: List[ParsedDocument] = []
    if not pdf_dir.is_dir():
        logger.error(f"PDF input directory not found or is not a directory: {pdf_dir}")
        return parsed_documents
    if not metadata_dir.is_dir():
        # Il est important d'avoir le répertoire de métadonnées même s'il est vide ou si certains fichiers manquent
        logger.warning(f"Metadata input directory not found or is not a directory: {metadata_dir}. Attempting to proceed; metadata might be missing for some PDFs.")
        # On pourrait créer le répertoire ici si l'intention est de toujours en avoir un, mais
        # pour l'instant, on logue un avertissement et on continue.
        # metadata_dir.mkdir(parents=True, exist_ok=True) # Optionnel

    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir} to parse.")

    for pdf_filepath in pdf_files:
        arxiv_id_from_filename = pdf_filepath.stem 
        
        text_content = parse_single_pdf(pdf_filepath)
        if text_content is None:
            logger.warning(f"Skipping document {arxiv_id_from_filename} due to PDF parsing error.")
            continue

        metadata_filepath = metadata_dir / f"{arxiv_id_from_filename}_metadata.json"
        metadata = load_metadata(metadata_filepath)
        
        document_metadata: Dict[str, Any]
        metadata_path_str: Optional[str]

        if metadata is None:
            logger.warning(f"Metadata file for {arxiv_id_from_filename} not found or failed to load from {metadata_filepath}. Proceeding with minimal metadata.")
            document_metadata = {"arxiv_id_inferred_from_filename": arxiv_id_from_filename}
            # Si des métadonnées sont absolument cruciales, on pourrait choisir de sauter le document ici.
            # Pour l'instant, on continue avec des métadonnées minimales.
            metadata_path_str = None
        else:
            document_metadata = metadata
            metadata_path_str = str(metadata_filepath)

        # S'assurer que l'ID ArXiv est présent, en priorité celui des métadonnées, sinon celui du nom de fichier
        final_arxiv_id = document_metadata.get("entry_id", arxiv_id_from_filename)
        # Normaliser l'ID ArXiv (enlever le suffixe de version s'il y en a un dans entry_id)
        if isinstance(final_arxiv_id, str):
            final_arxiv_id = final_arxiv_id.split("/")[-1].split("v")[0]
        else: # Fallback si entry_id n'est pas une chaîne attendue
            final_arxiv_id = arxiv_id_from_filename


        parsed_doc: ParsedDocument = {
            "arxiv_id": final_arxiv_id,
            "text_content": text_content,
            "metadata": document_metadata, # Contient toutes les métadonnées chargées du JSON
            "pdf_path": str(pdf_filepath),
            "metadata_path": metadata_path_str
        }
        parsed_documents.append(parsed_doc)
        logger.info(f"Successfully processed and added document: {parsed_doc['arxiv_id']}")

    logger.info(f"Finished parsing collection. Successfully processed {len(parsed_documents)} documents.")
    return parsed_documents


if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging(level="INFO")

    logger.info("Starting document parser test run (with modified path handling)...")

    # Pour tester, ces répertoires devraient exister et contenir des fichiers
    # créés par un appel précédent à arxiv_downloader.py (ou son test).
    # Nous utilisons les mêmes noms de répertoires de test que dans arxiv_downloader.py pour la cohérence.
    test_base_data_dir = Path(settings.DATA_DIR) / "corpus" / "test_parser_corpus" # Nom différent pour isoler
    test_pdf_input_dir = test_base_data_dir / "pdfs"
    test_metadata_input_dir = test_base_data_dir / "metadata"

    # Créer des fichiers factices pour le test si nécessaire (similaire à ce que arxiv_downloader ferait)
    # Normalement, ces fichiers seraient créés par l'étape de téléchargement.
    test_pdf_input_dir.mkdir(parents=True, exist_ok=True)
    test_metadata_input_dir.mkdir(parents=True, exist_ok=True)
    
    # Créer un PDF factice (le contenu réel du PDF n'est pas crucial pour tester le flux du parser)
    # PyMuPDF peut ouvrir des fichiers vides, mais cela pourrait causer des erreurs.
    # Un vrai petit PDF serait mieux, mais pour ce test, on va juste créer le fichier.
    # Pour un test robuste, il faudrait un vrai PDF.
    # Ici, on suppose que arxiv_downloader a déjà créé des fichiers.
    # Si vous exécutez ce test isolément, assurez-vous d'avoir des fichiers .pdf et .json dans les répertoires de test.
    # Exemple :
    # (test_pdf_input_dir / "testdoc01.pdf").write_text("dummy pdf content")
    # (test_metadata_input_dir / "testdoc01_metadata.json").write_text(json.dumps({"title": "Test Doc 01", "entry_id": "testdoc01v1"}))


    if not any(test_pdf_input_dir.iterdir()):
        logger.warning(f"No PDF files found in the test directory: {test_pdf_input_dir}.")
        logger.warning("Please ensure `arxiv_downloader.py` has run and populated a similar test directory,")
        logger.warning("or manually add test PDF and metadata JSON files to the paths above for a meaningful test.")
        logger.warning("For example, create 'testdoc01.pdf' and 'testdoc01_metadata.json'.")
    else:
        logger.info(f"Attempting to parse documents from PDF dir: {test_pdf_input_dir} and Metadata dir: {test_metadata_input_dir}")
        documents = parse_document_collection(
            pdf_dir=test_pdf_input_dir,
            metadata_dir=test_metadata_input_dir
        )
        if documents:
            logger.info(f"Successfully parsed {len(documents)} documents in test run.")
            for i, doc in enumerate(documents[:2]): 
                logger.info(f"--- Document {i+1} ---")
                logger.info(f"  ArXiv ID: {doc['arxiv_id']}")
                logger.info(f"  Title (from metadata): {doc['metadata'].get('title', 'N/A')}")
                logger.info(f"  Text Snippet: {doc['text_content'][:200].replace(chr(10), ' ')}...")
                logger.info(f"  PDF Path: {doc['pdf_path']}")
                logger.info(f"  Metadata Path: {doc['metadata_path']}")
        else:
            logger.warning("No documents were successfully parsed in the test run. Ensure test files are correctly placed and readable.")
    logger.info("--- Document parser test run finished ---")