"""Check a running local API, including one real Gemini answer.

Run from paper-rag: python scripts/check_api.py
The check uploads a sample and keeps its new index for inspection in /documents.
"""

import argparse
import json
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--check-refusal", action="store_true", help="Make a second Gemini request for an unsupported question")
    args = parser.parse_args()
    sample = Path(__file__).resolve().parents[1] / "data/samples/1706.03762v7.pdf"

    with httpx.Client(base_url=args.base_url, timeout=180, trust_env=False) as client:
        health = client.get("/health")
        health.raise_for_status()
        assert health.json() == {"status": "ok"}
        print("Health: OK", flush=True)

        with sample.open("rb") as file:
            uploaded = client.post("/upload", files={"file": (sample.name, file, "application/pdf")})
        uploaded.raise_for_status()
        assert uploaded.status_code == 201
        document = uploaded.json()
        print("Uploaded:", json.dumps(document, indent=2), flush=True)

        listing = client.get("/documents")
        listing.raise_for_status()
        assert any(item["document_id"] == document["document_id"] for item in listing.json())

        question = "Why are attention dot products divided by the square root of the key dimension?"
        payload = {"document_id": document["document_id"], "question": question, "top_k": 4}
        response = client.post("/ask", json=payload)
        response.raise_for_status()
        result = response.json()
        print("Answer:", json.dumps(result, indent=2, ensure_ascii=True), flush=True)
        assert result["answerable"] and result["cited_sources"], "Inspect the answer: expected supporting citations"
        for source in result["cited_sources"]:
            assert source["document_id"] == document["document_id"]
            assert source in result["retrieved_sources"]

        invalid = client.post("/ask", json={**payload, "question": "   "})
        assert invalid.status_code == 422
        missing = client.post("/ask", json={**payload, "document_id": "does-not-exist"})
        assert missing.status_code == 404
        print("Input validation: OK (422 and 404)", flush=True)

        if args.check_refusal:
            response = client.post("/ask", json={**payload, "question": "What was the weather in Yerevan when this paper was written?"})
            response.raise_for_status()
            result = response.json()
            print("Unsupported question:", result["answer"], flush=True)
            assert result["answerable"] is False, "Inspect the answer: expected refusal"
        print("API checks passed. The uploaded index remains available.")


if __name__ == "__main__":
    main()
