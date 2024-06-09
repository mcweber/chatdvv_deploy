# ---------------------------------------------------
# Version: 07.06.2024
# Author: M. Weber
# ---------------------------------------------------
# 07.06.2024 implemented rag with fulltext search
# ---------------------------------------------------

import chatdvv_module as ai

def main() -> None:

    ai.os.system('cls' if ai.os.name == 'nt' else 'clear')

    while True:

        print("-"*50)
        print(f"Anzahl Artikel: {ai.collection.count_documents({})}")
        print(f"Anzahl Artikel ohne Abstract: {ai.collection.count_documents({'ki_abstract': ''})}")
        print(f"Anzahl Artikel ohne Embeddings: {ai.collection.count_documents({'embeddings': {}})}")
        print("-"*50)
        print("[T]extsuche [V]ektorsuche [C]lear screen [S]tatisiken E[x]it")
        print("[G]enerate abstracts C[r]eate Embeddings")
        print("-"*50)
        befehl = input("Stelle eine Frage: ")
        print("-"*50)

        if befehl.upper() == "X":
            break

        elif befehl.upper() == "C":
            ai.os.system('cls' if ai.os.name == 'nt' else 'clear')

        elif befehl.upper() == "G":
            for i in range(100):
                print("*"*70)
                print(f"Generating abstracts {i}/10")
                print("*"*70)
                ai.generate_abstracts(input_field="text", output_field="ki_abstract", max_iterations=100)
                i += 1

        elif befehl.upper() == "R":
            ai.generate_embeddings(input_field="ki_abstract", output_field="embeddings", max_iterations=1000)

        elif befehl.upper() == "V":
            result = ai.vector_search_artikel(input("Vektor Suche: "))
            ai.print_results(result, 10)

        elif befehl.upper() == "T":
            cursor, count = ai.text_search(input("Text Suche: "))
            ai.print_results(cursor, 10)

        elif befehl.upper() == "S":
            print("Statistiken:")
            print(ai.group_by_field())
            print("-"*50)
            print(ai.list_fields())
            input("\nWeiter mit Enter...")

        else:
            # results = ai.vector_search(befehl, 3)
            results, results_count = ai.text_search(befehl)

            results_str = ""
            for result in results:
                print(f"[{result['datum']}] {result['titel'][:70] + '...'}")
                results_str += f"Datum: {result['datum']}\nArtikel: {result['text'][:500]}\n\n"

            print("-"*50)

            summary = ai.ask_llm(
                llm="openai",
                temperature=0.2,
                question=befehl,
                history=[],
                systemPrompt="Du bist ein hilfreicher Assistent und gibst Informationen aus dem Bereich Transport und Logistik.",
                results_str=results_str
                )

            print(summary)


if __name__ == "__main__":
    main()
