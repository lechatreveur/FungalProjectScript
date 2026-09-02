class GroundTruthCorrectorError(Exception):
    """Base exception for Ground Truth Corrector."""
    pass

class NotFoundError(GroundTruthCorrectorError):
    pass

class RevisionConflict(GroundTruthCorrectorError):
    pass

class DataIntegrityError(GroundTruthCorrectorError):
    pass

class SecurityError(GroundTruthCorrectorError):
    pass
