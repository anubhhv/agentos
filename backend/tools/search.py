import os
import httpx
from bs4 import BeautifulSoup

SERPER_KEY = os.getenv("SERPER_API_KEY", "")

async def web_search(query: str, num: int = 6) -> dict:
    """Search the live web via Serper API and return structured results."""
    if not SERPER_KEY:
        return {"error": "SERPER_API_KEY not set", "results": []}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
                json={"q": query, "num": num}
            )
            r.raise_for_status()
            data = r.json()

        results = []
        for item in data.get("organic", []):
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })

        answer_box = data.get("answerBox", {})
        knowledge = data.get("knowledgeGraph", {})

        return {
            "query": query,
            "results": results,
            "answer_box": answer_box if answer_box else None,
            "knowledge_graph": knowledge if knowledge else None,
            "total": len(results)
        }
    except Exception as e:
        return {"error": str(e), "results": []}


async def web_fetch(url: str) -> dict:
    """Fetch and extract readable text content from a URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AgentOS/1.0)"}
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            r = await client.get(url, headers=headers)
            r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
            tag.decompose()

        # Try to grab main content
        main = soup.find("main") or soup.find("article") or soup.find(id="content") or soup.body
        text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)

        # Collapse blank lines
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        clean = "\n".join(lines)

        title = soup.title.string.strip() if soup.title else url
        return {
            "url": url,
            "title": title,
            "content": clean[:8000],   # cap at 8k chars
            "length": len(clean)
        }
    except Exception as e:
        return {"error": str(e), "url": url, "content": ""}
