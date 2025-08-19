from src.search import hybrid_search
from src.generator import generate_answer
from src.config import INDEX_ALL
from src.search1 import build_filtered_context
from src.context_filter import build_filtered_context_highlights     


if __name__ == "__main__":
    query = 'În ce an a murit Mihai Eminescu?'
    print(f"Întrebare: {query}\n")

    # Use the updated hybrid_search function
    results = hybrid_search(query, index_name=INDEX_ALL, k=3)
    # results = hybrid_search1(query, index_name=INDEX_ALL, k=3)


    print("Cele mai relevante rezultate (hybrid search):\n")
    for result in results:
        doc_type = result["type"]
        name = result["name"]
        desc = result["description"]
        score = result["score"]

        # Print the enriched results
        print(f"[{doc_type.upper()}] {name} (score: {score:.4f})\n  {desc[:300]}...\n")

    # Construct the context for the LLM
    # context = "\n\n".join(
    #     f"[{result['type'].upper()}] {result['name']}:\n{result['description']}\n"
    #     # + (f"Keywords: {', '.join(result['keywords'])}\n" if result['keywords'] else "")
    #     + (f"Professions: {', '.join(result['professions'])}\n" if result['type'] == "author" and result['professions'] else "")
    #     + (f"Writings: {result['writings']}\n" if result['type'] == "author" and result['writings'] else "")
    #     + (f"Category: {result['category']}\n" if result['type'] == "publication" and result['category'] else "")
    #     for result in results
    # )

    # context = build_filtered_context(results, query, top_n_sentences=7)

    context = build_filtered_context_highlights(results, query, top_n_sentences=7)


    print("\n>>> Context extras:\n")
    print(context)

    # Generate an answer using the context
    answer = generate_answer(query, context, model="qwen2.5:72b")
    print("\n>>> Răspuns generat:\n")
    print(answer)
