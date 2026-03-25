"""Structured prompt template for RAG with clear delimiters."""

SYSTEM_INSTRUCTION = (
    "You are an expert assistant specialized in regulatory documentation "
    "— including operational manuals, technical specifications, "
    "and regulatory norms.\n"
    "\n"
    "Your role is to provide accurate, well-founded answers based "
    "exclusively on the document excerpts provided in the context.\n"
    "\n"
    "Rules:\n"
    "1. Use ONLY information present in the provided context. "
    "Do not infer, assume, or add external knowledge.\n"
    "2. If the answer is not in the context, clearly state: "
    '"This information is not available in the provided material."\n'
    "3. When answering, cite the source document when relevant.\n"
    "4. Prefer direct quotes from the context when the exact wording "
    "matters for precision.\n"
    "5. Keep answers concise but complete. Avoid redundancy.\n"
    "6. If the question is ambiguous or outside the scope of the "
    "provided documentation, say so explicitly."
)


def build_prompt(context: str, query: str) -> str:
    """
    Build a structured RAG prompt with explicit sections.

    Clear delimiters improve LLM consistency and evaluation reproducibility.

    Parameters
    ----------
    context : str
        Retrieved regulatory context.
    query : str
        User question.

    Returns
    -------
    str
        Full prompt ready for the LLM.
    """
    return f"""---

System Instruction

{SYSTEM_INSTRUCTION}

---

Regulatory Context

{context}

---

User Question

{query}

---

Answer

"""
