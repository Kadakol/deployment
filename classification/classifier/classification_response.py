from pydantic import BaseModel


class ClassificationResponse(BaseModel):

    filename: str
    content_type: str
    predicted_class: str
    confidence: float