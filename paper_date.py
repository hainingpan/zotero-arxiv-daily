import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote
import time
from typing import Union, List
import arxiv
from tqdm import trange,tqdm
from paper import ArxivPaper


def get_arxiv_date(
    category: Union[str, List[str]], 
    start_date: str, 
    end_date: str, 
    max_results: int = 500
) -> List[str]:
    """
    Fetch arXiv paper numbers for given categories and date range.
    
    Args:
        category: Single category string (e.g., 'quant-ph') or list of categories
        start_date: Start date in YYYYMMDD format (e.g., '20240101')
        end_date: End date in YYYYMMDD format (e.g., '20241231')
        max_results: Maximum number of results per API call (default: 1000)
    
    Returns:
        List of arXiv paper numbers (e.g., ['2401.12345', '2401.12346', ...])
    """
    
    # Convert single category to list for uniform processing
    print(category)
    print(start_date)
    print(end_date)
    print(max_results)
    if isinstance(category, str):
        categories = [category]
    else:
        categories = category
    
    all_arxiv_numbers = []
    
    # Loop through each category
    for cat in categories:
        print(f"Fetching papers for category: {cat}")
        
        base_url = 'http://export.arxiv.org/api/query'
        
        # Construct date range query
        date_query = f"submittedDate:[{start_date}* TO {end_date}*]"
        query = f"cat:{cat} AND {date_query}"
        
        start = 0
        
        while True:
            # Construct API URL
            url = f"{base_url}?search_query={quote(query)}&start={start}&max_results={max_results}"
            
            try:
                response = requests.get(url)
                response.raise_for_status()
                
                # Parse XML response
                root = ET.fromstring(response.content)
                
                # Find all paper entries
                entries = root.findall('{http://www.w3.org/2005/Atom}entry')
                
                if not entries:
                    break
                
                # Extract arXiv numbers from entries
                for entry in entries:
                    id_element = entry.find('{http://www.w3.org/2005/Atom}id')
                    if id_element is not None:
                        # Extract arXiv number from URL (e.g., from 'http://arxiv.org/abs/2401.12345v1')
                        arxiv_id = id_element.text.split('/')[-1]
                        # Remove version number if present (e.g., 'v1', 'v2')
                        if 'v' in arxiv_id:
                            arxiv_id = arxiv_id.split('v')[0]
                        all_arxiv_numbers.append(arxiv_id)
                
                print(f"  Retrieved {len(entries)} papers (total so far: {len(all_arxiv_numbers)})")
                
                # Update start position for next batch
                start += len(entries)
                
                # Rate limiting - be respectful to arXiv servers
                time.sleep(3)
                
                # If we got fewer results than requested, we've reached the end
                if len(entries) < max_results:
                    break
                    
            except requests.RequestException as e:
                print(f"Error fetching data for category {cat}: {e}")
                break
            except ET.ParseError as e:
                print(f"Error parsing XML response for category {cat}: {e}")
                break
    
    # Remove duplicates (in case a paper appears in multiple categories)
    unique_arxiv_numbers = list(set(all_arxiv_numbers))
    
    print(f"\nTotal unique papers found: {len(unique_arxiv_numbers)}")
    client = arxiv.Client(num_retries=10,delay_seconds=10)
    papers = []
    all_paper_ids = sorted(unique_arxiv_numbers)
    bar = tqdm(total=len(all_paper_ids),desc="Retrieving Arxiv papers")
    for i in range(0,len(all_paper_ids),50):
        search = arxiv.Search(id_list=all_paper_ids[i:i+50])
        batch = [ArxivPaper(p) for p in client.results(search)]
        bar.update(len(batch))
        papers.extend(batch)
    bar.close()

    return papers


# Example usage:
# if __name__ == "__main__":
#     # Single category example
#     quant_papers = get_arxiv_date(
#         category='quant-ph',
#         start_date='20250101',
#         end_date='20250131',
#         max_results=500
#     )
    
#     print(f"Found {len(quant_papers)} quantum physics papers")
#     print("First 10 papers:", quant_papers[:10])
    
    # # Multiple categories example
    # multi_papers = get_arxiv_date(
    #     category=['quant-ph', 'cond-mat.mes-hall', 'physics.optics'],
    #     start_date='20240601',
    #     end_date='20240630',
    #     max_results=200
    # )
    
    # print(f"\nFound {len(multi_papers)} papers across multiple categories")
    # print("First 10 papers:", multi_papers[:10])