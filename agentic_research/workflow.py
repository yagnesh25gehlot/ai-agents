from agents.planner import PlannerAgent
from agents.researcher import ResearchAgent
from agents.verifier import VerifierAgent
from agents.synthesizer import SynthesizerAgent


class ResearchWorkflow:
    def __init__(self):
        self.planner = PlannerAgent()
        self.researcher = ResearchAgent()
        self.verifier = VerifierAgent()
        self.synthesizer = SynthesizerAgent()

    def run(self, topic: str) -> str:
        plan = self.planner.create_plan(topic)

        verified_results = []

        for question in plan:
            research = self.researcher.research(question)

            if self.verifier.verify(research):
                verified_results.append(research)

        return self.synthesizer.synthesize(verified_results)
