"""Stub for a tool-selection router model."""

class Router:
    """Placeholder router that will learn to pick pipelines/tools."""

    def __init__(self, pipeline_ids=None):
        self.pipeline_ids = pipeline_ids or []

    def predict(self, question: str):
        raise NotImplementedError("Router.predict is not implemented yet.")
