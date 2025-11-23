import asyncio
import aiohttp
from bs4 import BeautifulSoup
import csv
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin
import logging
from datetime import datetime
import re
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TvSeansScraperConfig:
    BASE_URL = "https://tvseans.az"
    MOVIES_URL = f"{BASE_URL}/az/category/movies"
    PER_PAGE = 30
    MAX_CONCURRENT_REQUESTS = 10
    REQUEST_TIMEOUT = 30
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 2
    CHECKPOINT_FILE = "scraper_checkpoint.json"
    MAX_PAGES = 500  # Maximum pages to scrape


class TvSeansScraperAdvanced:
    def __init__(self):
        self.config = TvSeansScraperConfig()
        self.semaphore = asyncio.Semaphore(self.config.MAX_CONCURRENT_REQUESTS)
        self.session: Optional[aiohttp.ClientSession] = None
        self.movies_data: List[Dict] = []
        self.scraped_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.total_movies = 0
        self.completed_movies = 0

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self.config.REQUEST_TIMEOUT)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def load_checkpoint(self) -> Dict:
        """Load previous scraping progress"""
        checkpoint_path = Path(self.config.CHECKPOINT_FILE)
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.scraped_urls = set(data.get('scraped_urls', []))
                    self.failed_urls = set(data.get('failed_urls', []))
                    logger.info(f"Loaded checkpoint: {len(self.scraped_urls)} scraped, {len(self.failed_urls)} failed")
                    return data
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
        return {}

    def save_checkpoint(self, movie_urls: List[str]):
        """Save current progress"""
        try:
            checkpoint = {
                'scraped_urls': list(self.scraped_urls),
                'failed_urls': list(self.failed_urls),
                'total_movies': self.total_movies,
                'completed_movies': self.completed_movies,
                'all_movie_urls': movie_urls,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.config.CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            logger.info(f"Checkpoint saved: {self.completed_movies}/{self.total_movies}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

    async def fetch_page(self, url: str, retry_count: int = 0) -> Optional[str]:
        """Fetch a page with retry logic"""
        async with self.semaphore:
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        logger.warning(f"Status {response.status} for {url}")
                        return None
            except asyncio.TimeoutError:
                if retry_count < self.config.RETRY_ATTEMPTS:
                    logger.warning(f"Timeout, retry {retry_count + 1}/{self.config.RETRY_ATTEMPTS} for {url}")
                    await asyncio.sleep(self.config.RETRY_DELAY * (retry_count + 1))
                    return await self.fetch_page(url, retry_count + 1)
                else:
                    logger.error(f"Timeout after {self.config.RETRY_ATTEMPTS} retries: {url}")
                    return None
            except Exception as e:
                if retry_count < self.config.RETRY_ATTEMPTS:
                    logger.warning(f"Error, retry {retry_count + 1}/{self.config.RETRY_ATTEMPTS} for {url}: {str(e)}")
                    await asyncio.sleep(self.config.RETRY_DELAY * (retry_count + 1))
                    return await self.fetch_page(url, retry_count + 1)
                else:
                    logger.error(f"Failed after {self.config.RETRY_ATTEMPTS} retries {url}: {str(e)}")
                    return None

    def extract_movie_links(self, html: str) -> List[str]:
        """Extract movie detail page links from listing page"""
        soup = BeautifulSoup(html, 'html.parser')
        links = []

        movie_cards = soup.find_all('div', class_='m_s')
        for card in movie_cards:
            link = card.find('a', {'data-pjax': '0'})
            if link and link.get('href'):
                full_url = urljoin(self.config.BASE_URL, link['href'])
                links.append(full_url)

        return links

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        text = text.replace('&nbsp;', ' ')
        text = text.replace('\xa0', ' ')
        return text

    def extract_movie_details(self, html: str, url: str) -> Dict:
        """Extract all details from a movie page"""
        soup = BeautifulSoup(html, 'html.parser')
        data = {
            'url': url,
            'title_az': '',
            'title_original': '',
            'year': '',
            'country': '',
            'age_rating': '',
            'duration_minutes': '',
            'languages': '',
            'quality': '',
            'genres': '',
            'rating_tvseans': '',
            'rating_imdb': '',
            'rating_kinopoisk': '',
            'description': '',
            'director': '',
            'screenwriter': '',
            'actors': '',
            'poster_url': '',
            'frame_urls': '',
            'comments_count': 0,
            'scraped_at': datetime.now().isoformat()
        }

        try:
            # Extract titles
            title_section = soup.find('div', class_='m-name')
            if title_section:
                # Get original title from span first
                original_span = title_section.find('span')
                if original_span:
                    data['title_original'] = self.clean_text(original_span.get_text())
                    # Remove span to get Azerbaijani title
                    original_span.decompose()

                # Now get Azerbaijani title (with span removed)
                title_az = self.clean_text(title_section.get_text())
                if title_az:
                    data['title_az'] = title_az

            # Extract year, country, age rating
            year_section = soup.find('div', class_='m-year')
            if year_section:
                year_text = year_section.get_text()

                # Extract year
                year_match = re.search(r'(\d{4})', year_text)
                if year_match:
                    data['year'] = year_match.group(1)

                # Split by comma
                parts = [p.strip() for p in year_text.split(',')]

                # Extract age rating (contains + and digits, like "18+", "16+", "12+")
                age_parts = [p for p in parts if re.search(r'\d+\+', p)]
                if age_parts:
                    data['age_rating'] = self.clean_text(age_parts[0])

                # Extract country (everything except year and age rating)
                country_parts = []
                for p in parts:
                    p_clean = self.clean_text(p)
                    # Skip if it's year or age rating
                    if not re.search(r'^\d{4}$', p_clean) and not re.search(r'\d+\+', p_clean):
                        country_parts.append(p_clean)

                if country_parts:
                    data['country'] = ', '.join(country_parts)

            # Extract duration, languages, quality
            video_info = soup.find('div', class_='video-cr')
            if video_info:
                info_divs = video_info.find_all('div')
                for div in info_divs:
                    text = div.get_text()

                    # Duration
                    if 'dəq' in text:
                        duration_match = re.search(r'(\d+)', text)
                        if duration_match:
                            data['duration_minutes'] = duration_match.group(1)

                    # Languages (uppercase indicates language codes)
                    elif re.search(r'\b[A-Z]{2}\b', text.upper()) and 'az' in text.lower():
                        data['languages'] = self.clean_text(text)

                    # Quality
                    elif any(q in text for q in ['1080p', '720p', '360p', '4K']):
                        data['quality'] = self.clean_text(text)

            # Extract genres
            genres_section = soup.find('div', class_='m-genres')
            if genres_section:
                genres = [self.clean_text(g.get_text()) for g in genres_section.find_all('span', itemprop='genre')]
                data['genres'] = ', '.join(genres)

            # Extract ratings
            adv_section = soup.find('div', class_='m-adv')
            if adv_section:
                adv_html = str(adv_section)

                # TvSeans rating
                tvseans_match = re.search(r'<b>([\d.]+)</b>\s*</span>\s*TvSeans', adv_html)
                if tvseans_match:
                    data['rating_tvseans'] = tvseans_match.group(1)

                # IMDb rating
                imdb_match = re.search(r'<b>([\d.]+)</b>\s*</span>\s*IMDb', adv_html)
                if imdb_match:
                    data['rating_imdb'] = imdb_match.group(1)

                # Kinopoisk rating
                kp_match = re.search(r'<b>([\d.]+)</b>\s*</span>\s*Kinopoisk', adv_html)
                if kp_match:
                    data['rating_kinopoisk'] = kp_match.group(1)

            # Extract description
            story_section = soup.find('div', class_='movie_story')
            if story_section:
                for br in story_section.find_all('br'):
                    br.replace_with(' ')
                description = story_section.get_text()
                # Remove the promotional text at the end
                description = re.sub(r'www\.TVSEANS\.com.*$', '', description, flags=re.IGNORECASE)
                description = re.sub(r'Film və serialları.*$', '', description, flags=re.IGNORECASE)
                data['description'] = self.clean_text(description)

            # Extract crew information
            persons_section = soup.find('div', class_='m-persons')
            if persons_section:
                dl = persons_section.find('dl')
                if dl:
                    # Director
                    dt_director = dl.find('dt', string=re.compile(r'Rejissor', re.IGNORECASE))
                    if dt_director:
                        dd = dt_director.find_next_sibling('dd')
                        if dd:
                            directors = [self.clean_text(a.get_text()) for a in dd.find_all('a')]
                            data['director'] = ', '.join(directors)

                    # Screenwriter
                    dt_writer = dl.find('dt', string=re.compile(r'Ssenarist', re.IGNORECASE))
                    if dt_writer:
                        dd = dt_writer.find_next_sibling('dd')
                        if dd:
                            writers = [self.clean_text(a.get_text()) for a in dd.find_all('a')]
                            data['screenwriter'] = ', '.join(writers)

                    # Actors
                    dt_actors = dl.find('dt', string=re.compile(r'Digər Rollarda|Rollarda', re.IGNORECASE))
                    if dt_actors:
                        dd = dt_actors.find_next_sibling('dd')
                        if dd:
                            actors = [self.clean_text(a.get_text()) for a in dd.find_all('a')
                                    if 'persons' not in a.get('href', '') and 'Daha çox' not in a.get_text()]
                            data['actors'] = ', '.join(actors)

            # Extract poster URL
            poster = soup.find('div', class_='poster')
            if poster:
                img = poster.find('img')
                if img and img.get('src'):
                    data['poster_url'] = urljoin(self.config.BASE_URL, img['src'])

            # Extract frame URLs
            frames_section = soup.find('div', class_='frames')
            if frames_section:
                frame_links = frames_section.find_all('a', {'data-fancybox': 'images'})
                frame_urls = [urljoin(self.config.BASE_URL, a['href']) for a in frame_links if a.get('href')]
                data['frame_urls'] = '|'.join(frame_urls)

            # Count comments
            comments = soup.find_all('div', class_='comment')
            data['comments_count'] = len(comments)

        except Exception as e:
            logger.error(f"Error extracting details from {url}: {str(e)}", exc_info=True)

        return data

    async def scrape_movie(self, url: str) -> Optional[Dict]:
        """Scrape a single movie page"""
        if url in self.scraped_urls:
            logger.debug(f"Skipping already scraped: {url}")
            return None

        logger.info(f"Scraping [{self.completed_movies + 1}/{self.total_movies}]: {url}")
        html = await self.fetch_page(url)

        if html:
            data = self.extract_movie_details(html, url)
            self.scraped_urls.add(url)
            self.completed_movies += 1
            return data
        else:
            self.failed_urls.add(url)
            self.completed_movies += 1
            return None

    async def scrape_all_movies(self):
        """Main scraping function with progress tracking"""
        logger.info("Starting advanced scraper with checkpoint support...")

        # Load previous progress
        checkpoint = self.load_checkpoint()

        # Collect all movie URLs
        all_movie_urls = checkpoint.get('all_movie_urls', [])

        if not all_movie_urls:
            logger.info("Collecting movie URLs from listing pages...")

            # Fetch first page
            first_page_html = await self.fetch_page(
                f"{self.config.MOVIES_URL}?page=1&per-page={self.config.PER_PAGE}"
            )
            if not first_page_html:
                logger.error("Failed to fetch first page")
                return

            first_page_links = self.extract_movie_links(first_page_html)
            all_movie_urls.extend(first_page_links)
            logger.info(f"Found {len(first_page_links)} movies on page 1")

            # Fetch remaining pages in batches
            batch_size = 20
            for batch_start in range(2, self.config.MAX_PAGES + 1, batch_size):
                batch_end = min(batch_start + batch_size, self.config.MAX_PAGES + 1)

                page_tasks = []
                for page in range(batch_start, batch_end):
                    url = f"{self.config.MOVIES_URL}?page={page}&per-page={self.config.PER_PAGE}"
                    page_tasks.append(self.fetch_page(url))

                logger.info(f"Fetching listing pages {batch_start}-{batch_end-1}...")
                pages_html = await asyncio.gather(*page_tasks)

                for page_num, html in enumerate(pages_html, start=batch_start):
                    if html:
                        links = self.extract_movie_links(html)
                        if links:
                            all_movie_urls.extend(links)
                            logger.info(f"Found {len(links)} movies on page {page_num}")
                        else:
                            logger.info(f"No movies found on page {page_num}, stopping pagination")
                            break

                # Small delay between batches
                await asyncio.sleep(1)

            # Remove duplicates
            all_movie_urls = list(set(all_movie_urls))
            logger.info(f"Total unique movies found: {len(all_movie_urls)}")

        self.total_movies = len(all_movie_urls)

        # Filter out already scraped URLs
        urls_to_scrape = [url for url in all_movie_urls if url not in self.scraped_urls]
        logger.info(f"Movies to scrape: {len(urls_to_scrape)} (already done: {len(self.scraped_urls)})")

        # Scrape movies in batches with checkpointing
        batch_size = 50
        for i in range(0, len(urls_to_scrape), batch_size):
            batch = urls_to_scrape[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(urls_to_scrape) + batch_size - 1) // batch_size}")

            movie_tasks = [self.scrape_movie(url) for url in batch]
            batch_results = await asyncio.gather(*movie_tasks)

            # Add valid results
            for result in batch_results:
                if result:
                    self.movies_data.append(result)

            # Save checkpoint after each batch
            self.save_checkpoint(all_movie_urls)

            # Save intermediate results
            if self.movies_data:
                self.save_to_csv('tvseans_movies_partial.csv')

        logger.info(f"Scraping complete! Successfully scraped {len(self.movies_data)} movies")
        logger.info(f"Failed URLs: {len(self.failed_urls)}")

    def save_to_csv(self, filename: str = 'tvseans_movies.csv'):
        """Save scraped data to CSV"""
        if not self.movies_data:
            logger.warning("No data to save")
            return

        fieldnames = [
            'url', 'title_az', 'title_original', 'year', 'country', 'age_rating',
            'duration_minutes', 'languages', 'quality', 'genres',
            'rating_tvseans', 'rating_imdb', 'rating_kinopoisk',
            'description', 'director', 'screenwriter', 'actors',
            'poster_url', 'frame_urls', 'comments_count', 'scraped_at'
        ]

        try:
            with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.movies_data)

            logger.info(f"Data saved to {filename} ({len(self.movies_data)} records)")
        except Exception as e:
            logger.error(f"Error saving to CSV: {str(e)}")


async def main():
    """Main entry point"""
    async with TvSeansScraperAdvanced() as scraper:
        await scraper.scrape_all_movies()
        scraper.save_to_csv('tvseans_movies.csv')

        # Log failed URLs if any
        if scraper.failed_urls:
            logger.warning(f"Failed to scrape {len(scraper.failed_urls)} URLs:")
            for url in list(scraper.failed_urls)[:10]:
                logger.warning(f"  - {url}")


if __name__ == "__main__":
    asyncio.run(main())
