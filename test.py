import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def extract_all_internal_links(base_url: str, max_depth=2) -> list[str]:
    visited = set()
    to_visit = [(base_url, 0)]
    domain = urlparse(base_url).netloc

    while to_visit:
        current_url, depth = to_visit.pop()
        if current_url in visited or depth > max_depth:
            continue
        visited.add(current_url)

        try:
            response = requests.get(current_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
        except Exception as e:
            print(f"Failed to access {current_url}: {e}")
            continue

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(current_url, href)
            parsed = urlparse(full_url)

            # Only follow internal links
            if parsed.netloc == domain:
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if clean_url not in visited:
                    to_visit.append((clean_url, depth + 1))

    return list(sorted(visited))

if __name__ == "__main__":
    base = "https://study.iitm.ac.in/ds/"
    urls = extract_all_internal_links(base, max_depth=11)  # increase depth if needed
    print(urls)
    print(len(urls))
