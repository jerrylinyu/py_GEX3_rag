from rag import answer as _rag_answer

# ask AI for help
class RAGTool:
    name = "university_it_policy_rag"
    description = (
        "Look up the university IT policies and answer a user's question. "
        "Use only for questions about printing, storage, accounts, VPN, "
        "Wi-Fi, and email at this university."
    )

    def run(self, question: str) -> str:
        if not question or not question.strip():
            return "Please provide a question."
        return _rag_answer(question.strip())


if __name__ == "__main__":
    tool = RAGTool()
    print(tool.run("Which Wi-Fi should guests use?"))
    print(tool.run("What is the exact phone number for the IT Service Desk?"))