from dotenv import load_dotenv
from search import answer

load_dotenv()


def main():

    print("Terminal Chat is ready!")

    while True:
        try:
            user_question = input("What is your question? ").strip()

            if user_question.lower() in ["exit", "quit", "sair"]:
                print("Goodbye!")
                break

            if not user_question:
                continue

            answer_template = answer(user_question)

            # Return the response to the user
            print(f"{answer_template}\n")

        except (KeyboardInterrupt, EOFError):
            # Gracefully handle Ctrl+C or Ctrl+D
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()