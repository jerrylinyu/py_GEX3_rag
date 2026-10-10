from rag_tool import RAGTool

# ask AI for help
class Agent:
    def __init__(self):
        self.tool = RAGTool()

    def handle(self, user_input: str) -> str:
        return self.tool.run(user_input)


def main():
    agent = Agent()
    test_questions = [
        "Which Wi-Fi should guests use?",
        "How long is an unreleased print job kept?",
        "What should I do if my account is locked?",
        "What exact phone number should I call for the University IT Service Desk?",
    ]
    for q in test_questions:
        print("=" * 70)
        print("Q:", q)
        print("A:", agent.handle(q))


if __name__ == "__main__":
    main()