from typing import Optional
from src.models import ResponseModel

class ResponseService:
    def __init__(self):
        self.descriptions = [
            'Successful check',
            'Person is not speaking',
            'No face present',
        ]

    def format_response(self, code: Optional[int] = None, result: float = 0.0) -> ResponseModel:
        """
        Format the response with appropriate code and score
        :param code: Response code
        :param result: Processing result
        :return: Formatted response
        """
        if code != 2:
            result = round((result > 0).mean() * 100, 2)
            code = int(result < 4.0)

        score = self._calculate_score(result)

        return ResponseModel(
            code=code,
            description=self.descriptions[code],
            result=result,
            score=score
        )

    def _calculate_score(self, result: float) -> float:
        """Calculate the score based on the result"""
        if result < 10:
            return round((result / 10) * 50, 2)
        return round(50 + (result - 10) * 5 / 9, 2) 