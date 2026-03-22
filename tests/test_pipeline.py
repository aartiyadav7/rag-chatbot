from src.pipeline.ingest_pipeline import ingest_url
from src.pipeline.query_pipeline import run_query

def test_full_pipeline():
    # Ingest
    ingest_url("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
    print("\n✅ Ingestion complete")

    # Query
    result = run_query("What is retrieval augmented generation?")
    assert result["answer"]
    assert len(result["sources"]) 

    print(f"\n✅ Query complete!")
    print(f"\n🤖 Answer:\n{result['answer']}")
    print(f"\n📄 Sources: {result['sources']}")