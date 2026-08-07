from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from flask import Response

class TrackingCorrectorError(Exception):
    """Base exception for tracking corrector application errors."""
    status_code: int = 500
    code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        if status_code is not None:
            self.status_code = status_code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": "error",
            "code": self.code,
            "message": self.message,
            "details": self.details
        }

class ValidationError(TrackingCorrectorError):
    status_code = 422
    code = "VALIDATION_ERROR"

class NotFoundError(TrackingCorrectorError):
    status_code = 404
    code = "NOT_FOUND"

class RevisionConflict(TrackingCorrectorError):
    status_code = 409
    code = "REVISION_CONFLICT"

class DataIntegrityError(TrackingCorrectorError):
    status_code = 400
    code = "DATA_INTEGRITY_ERROR"

class AccessDeniedError(TrackingCorrectorError):
    status_code = 403
    code = "ACCESS_DENIED"

def handle_app_error(error: TrackingCorrectorError) -> tuple["Response", int]:
    """Flask error handler for custom application exceptions."""
    from flask import jsonify

    return jsonify(error.to_dict()), error.status_code
