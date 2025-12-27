from workflow import ResearchWorkflow


if __name__ == "__main__":
    topic = "Distributed Systems Consistency Models"

    workflow = ResearchWorkflow()
    result = workflow.run(topic)

    print("\n" + "=" * 80)
    print(result)
    print("=" * 80)
