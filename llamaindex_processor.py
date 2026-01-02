from llama_cloud_services import LlamaExtract
from llama_cloud import ExtractConfig, ExtractMode
from pydantic import BaseModel, Field
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()


model_name = "document-classifier"


class ClassificationSchema(BaseModel):
    document_type: str = Field(description="Document type: invoice, receipt, purchase_order, etc.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")

def _normalize_document_type(raw_type: str) -> str:
        """Normalize document type to standard format

        Args:
            raw_type: Raw document type from provider

        Returns:
            Normalized document type (lowercase, underscores)
        """
        # Convert to lowercase and replace spaces/hyphens with underscores
        normalized = raw_type.lower().strip()
        normalized = normalized.replace(" ", "_").replace("-", "_")

        # Map common variations to standard types
        type_mappings = {
            "invoices": "invoice",
            "receipts": "receipt",
            "purchase_orders": "purchase_order",
            "po": "purchase_order",
            "contracts": "contract",
            "agreement": "contract",
            "bank_statements": "bank_statement",
            "statement": "bank_statement",
            "expense_reports": "expense_report",
            "expenses": "expense_report",
            "credit_memos": "credit_memo",
            "credit_note": "credit_memo",
            "packing_slips": "packing_slip",
            "delivery_note": "packing_slip",
            "quotes": "quote",
            "quotation": "quote",
            "estimate": "quote",
            "timesheets": "timesheet",
            "time_card": "timesheet",
            "tax_forms": "tax_form",
            "tax_document": "tax_form",
            "minutes": "meeting_minutes",
            "meeting_notes": "meeting_minutes",
            "letter": "correspondence",
            "email": "correspondence",
            "memo": "correspondence",
        }

        # Return mapped type or original if valid
        if normalized in type_mappings:
            return type_mappings[normalized]

        # Default to correspondence for unknown types
        return "correspondence"



async def classify_document(document_content: bytes,file_type):
    """Classify document and return type with confidence"""

    api_key = os.getenv("LLAMA_API_KEY")
    extractor = LlamaExtract(api_key=api_key)

    try:
        agent = extractor.get_agent(name=model_name)
    except:
        config = ExtractConfig(extraction_mode=ExtractMode.BALANCED, confidence_scores=True)
        agent = extractor.create_agent(name=model_name, data_schema=ClassificationSchema, config=config)

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
        tmp.write(document_content)
        tmp_path = tmp.name

    try:
        result = await agent.aextract(tmp_path)
        data = result.data
        document_type = _normalize_document_type(data.get("document_type", "unclassified"))

        return document_type, data.get("confidence", 0.0)

    finally:
        os.unlink(tmp_path)