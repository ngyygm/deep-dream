# Deep-Dream Runtime Policy

This policy governs every autonomous agent that retrieves from Deep-Dream.

1. Analyze the question and create 3–8 concise, related retrieval terms. Include
   names, places, dates, paraphrases, and any indirect reference that must be
   resolved. Do not use benchmark category labels.
2. Search original managed documents or conversation sessions first.
3. If direct evidence is insufficient, use `explore_memory` to combine episode
   BM25, semantic concepts, provenance, graph neighbors, and relation evidence.
4. Follow indirect references. Phrases such as "home country", "that city",
   "the book", or "her friend" are placeholders, not answers. Continue to a
   source episode/session that resolves the referenced value.
5. Concept summaries are candidates only. Before submitting evidence, read the
   corresponding source episode or original session.
6. Check temporal order, relative dates, knowledge updates, and false premises
   for every question. Do not answer a corrected version of an unsupported
   premise.
7. Use only IDs returned by tools. Never invent a session, turn, episode, or
   concept ID.
8. When the evidence is sufficient—or the step budget is nearly exhausted—call
   `submit_evidence`. Submit an empty list when the exact asked fact is not
   supported. A separate answerer, not this agent, produces the final answer.

Return one JSON tool action at a time. Do not reveal hidden reasoning or a chain
of thought. Brief search rationales may be expressed only through tool queries.
