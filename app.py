from flask import Flask, render_template, request
from src.search import hybrid_search
from src.generator import generate_answer
from src.config import INDEX_ALL
from src.context_filter import build_filtered_context_highlights 

app = Flask(__name__)
chat_history = []  # each item will be: {"question": ..., "answer": ...}

@app.route("/", methods=["GET", "POST"])
def index():
    global chat_history

    if request.method == "POST":
        if "clear" in request.form:
            chat_history.clear()
        else:
            query = request.form["question"]
            model = request.form.get("model", "gemma3:12b")

            results = hybrid_search(query, index_name=INDEX_ALL, k=3)
            context = build_filtered_context_highlights(results, query, top_n_sentences=7)
            answer = generate_answer(query, context, model=model).strip()
            # print used model
            print(f"Used model: {model}")

            chat_history.insert(0, {
                "question": query,
                "answer": answer
            })

    return render_template("index.html", history=chat_history)


if __name__ == "__main__":
    app.run(debug=True)
